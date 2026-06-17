from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from quant_rabbit.cli import main
from quant_rabbit.self_improvement_audit import (
    PROFITABILITY_DISCIPLINE_CODES,
    PROJECTION_PENDING_EXPIRY_GRACE_SECONDS,
    STATUS_ACTION_REQUIRED,
    STATUS_BLOCKED,
    SelfImprovementAuditor,
    _effect_metrics,
    _directional_forecast_invalidation_first_like,
    _directional_forecast_target_timeout_like,
    _gateway_close_recovery_observation,
    _intent_live_readiness_family_breakdown,
    _normalized_pending_order_type,
    _profitability_findings,
    _projection_expired,
    _report_perspective_alignment_text,
    _top_intent_blockers,
    _top_intent_live_readiness_blockers,
)
from quant_rabbit.paths import DEFAULT_EXECUTION_LEDGER_DB, DEFAULT_SELF_IMPROVEMENT_HISTORY_DB


class SelfImprovementAuditorTest(unittest.TestCase):
    def test_perspective_alignment_report_keeps_later_opposite_rail_view(self) -> None:
        text = _report_perspective_alignment_text(
            {
                "status": "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED",
                "range_forecast_method_mismatch_groups": 4,
                "range_forecast_method_mismatch_lanes": 9,
                "range_forecast_method_mismatch_top": [
                    {"pair": "AUD_JPY", "direction": "LONG", "method_mismatch_lanes": 3, "range_rotation_lanes": 0},
                    {"pair": "AUD_JPY", "direction": "SHORT", "method_mismatch_lanes": 3, "range_rotation_lanes": 0},
                    {"pair": "USD_CHF", "direction": "LONG", "method_mismatch_lanes": 2, "range_rotation_lanes": 1},
                    {
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "method_mismatch_lanes": 1,
                        "range_rotation_lanes": 1,
                        "range_rotation_other_side_lanes": 2,
                        "range_rotation_other_side_directions": [{"code": "SHORT", "count": 2}],
                        "range_rotation_other_side_top_live_blocker_codes": [
                            {"code": "SPREAD_TOO_WIDE", "count": 2}
                        ],
                    },
                ],
            }
        )

        self.assertIn("USD_CAD LONG mismatch=1", text)
        self.assertIn("other_rail=SHORT", text)
        self.assertIn("other_blockers=SPREAD_TOO_WIDE", text)

    def test_default_history_db_is_separate_from_execution_ledger(self) -> None:
        auditor = SelfImprovementAuditor()

        self.assertEqual(auditor.db_path, DEFAULT_EXECUTION_LEDGER_DB)
        self.assertEqual(auditor.history_db_path, DEFAULT_SELF_IMPROVEMENT_HISTORY_DB)
        self.assertNotEqual(auditor.history_db_path, auditor.db_path)

    def test_pending_order_type_normalization_matches_broker_and_intents(self) -> None:
        self.assertEqual(_normalized_pending_order_type("LIMIT_ORDER"), "LIMIT")
        self.assertEqual(_normalized_pending_order_type("STOP"), "STOP-ENTRY")
        self.assertEqual(_normalized_pending_order_type("STOP_ORDER"), "STOP-ENTRY")
        self.assertEqual(_normalized_pending_order_type("STOP_ENTRY"), "STOP-ENTRY")
        self.assertEqual(_normalized_pending_order_type("MARKET_IF_TOUCHED_ORDER"), "MARKET-IF-TOUCHED")

    def test_projection_expiry_uses_live_telemetry_grace(self) -> None:
        grace = timedelta(seconds=PROJECTION_PENDING_EXPIRY_GRACE_SECONDS)
        row = {
            "timestamp_emitted_utc": (
                _NOW - timedelta(minutes=30) - grace + timedelta(seconds=1)
            ).isoformat(),
            "resolution_window_min": 30.0,
        }
        self.assertFalse(_projection_expired(row, now=_NOW))

        row["timestamp_emitted_utc"] = (
            _NOW - timedelta(minutes=30) - grace - timedelta(seconds=1)
        ).isoformat()
        self.assertTrue(_projection_expired(row, now=_NOW))

    def test_no_touch_directional_miss_is_not_invalidation_first(self) -> None:
        no_touch = {
            "direction": "UP",
            "resolution_status": "MISS",
            "resolution_evidence": (
                "target 1.10200 not reached; invalidation 1.09900 also untouched in forecast window"
            ),
        }
        invalidation_first = {
            "direction": "UP",
            "resolution_status": "MISS",
            "resolution_evidence": "2026-06-16T00:10:00Z invalidation 1.09900 touched before target 1.10200",
        }

        self.assertFalse(_directional_forecast_invalidation_first_like(no_touch))
        self.assertTrue(_directional_forecast_target_timeout_like(no_touch))
        self.assertTrue(_directional_forecast_invalidation_first_like(invalidation_first))
        self.assertFalse(_directional_forecast_target_timeout_like(invalidation_first))

    def test_top_intent_blockers_ignore_dry_run_strategy_warnings(self) -> None:
        blockers = _top_intent_blockers(
            {
                "results": [
                    {
                        "lane_id": "trend_trader:EUR_AUD:LONG:TREND_CONTINUATION",
                        "status": "LIVE_READY",
                        "risk_issues": [],
                        "strategy_issues": [
                            {
                                "code": "STRATEGY_PROFILE_MISSING",
                                "message": "forecast-seeded advisory profile warning",
                                "severity": "WARN",
                            }
                        ],
                        "live_strategy_issues": [
                            {
                                "code": "STRATEGY_PROFILE_MISSING",
                                "message": "forecast-seeded advisory profile warning",
                                "severity": "WARN",
                            }
                        ],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                        "status": "DRY_RUN_BLOCKED",
                        "risk_issues": [
                            {
                                "code": "REWARD_RISK_TOO_LOW",
                                "message": "reward/risk below floor",
                                "severity": "BLOCK",
                            }
                        ],
                        "strategy_issues": [],
                        "live_strategy_issues": [],
                        "live_blockers": ["EUR_USD SHORT forecast confidence below live floor"],
                    },
                ]
            }
        )

        messages = {item["message"] for item in blockers}
        self.assertNotIn("STRATEGY_PROFILE_MISSING", messages)
        self.assertIn("REWARD_RISK_TOO_LOW", messages)
        self.assertIn("EUR_USD SHORT forecast confidence below live floor", messages)

    def test_live_readiness_blockers_include_warn_live_gates_for_dry_run_passed(self) -> None:
        blockers = _top_intent_live_readiness_blockers(
            {
                "results": [
                    {
                        "lane_id": "failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT",
                        "status": "DRY_RUN_PASSED",
                        "risk_issues": [
                            {
                                "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                "message": "forecast confidence below live floor",
                                "severity": "WARN",
                            },
                            {
                                "code": "SPREAD_ADVISORY",
                                "message": "spread is elevated but still below block floor",
                                "severity": "WARN",
                            },
                        ],
                        "strategy_issues": [
                            {
                                "code": "STRATEGY_PROFILE_MISSING",
                                "message": "dry-run advisory profile warning",
                                "severity": "WARN",
                            }
                        ],
                        "live_strategy_issues": [
                            {
                                "code": "STRATEGY_NOT_ELIGIBLE",
                                "message": "strategy profile is not live eligible",
                                "severity": "WARN",
                                "strategy_profile_evidence": {
                                    "profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                    "required_fix": "both live and pretrade feedback are negative",
                                    "live_net_jpy": -1501.57,
                                    "pretrade_net_jpy": -837.76,
                                },
                            }
                        ],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "failure_trader:GBP_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                        "status": "DRY_RUN_PASSED",
                        "risk_issues": [],
                        "strategy_issues": [],
                        "live_strategy_issues": [],
                        "live_blockers": ["legacy live blocker fallback"],
                    },
                    {
                        "lane_id": "range_trader:EUR_CHF:SHORT:RANGE_ROTATION",
                        "status": "LIVE_READY",
                        "risk_issues": [
                            {
                                "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                "message": "live-ready advisory must not be counted",
                                "severity": "WARN",
                            }
                        ],
                        "live_strategy_issues": [],
                        "live_blockers": [],
                    },
                ]
            },
            statuses={"DRY_RUN_PASSED"},
        )

        messages = {item["message"]: item for item in blockers}
        self.assertEqual(messages["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]["count"], 1)
        self.assertEqual(messages["STRATEGY_NOT_ELIGIBLE"]["count"], 1)
        self.assertEqual(messages["legacy live blocker fallback"]["count"], 1)
        self.assertNotIn("SPREAD_ADVISORY", messages)
        self.assertNotIn("STRATEGY_PROFILE_MISSING", messages)

    def test_live_readiness_family_breakdown_separates_repair_surfaces(self) -> None:
        breakdown = _intent_live_readiness_family_breakdown(
            {
                "results": [
                    {
                        "lane_id": "failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT",
                        "status": "DRY_RUN_PASSED",
                        "intent": {
                            "pair": "AUD_CAD",
                            "side": "SHORT",
                            "order_type": "LIMIT",
                            "market_context": {"method": "BREAKOUT_FAILURE"},
                            "metadata": {"forecast_direction": "DOWN", "forecast_confidence": 0.49},
                        },
                        "risk_metrics": {"reward_risk": 1.4},
                        "risk_issues": [
                            {
                                "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                "message": "forecast confidence below live floor",
                                "severity": "WARN",
                            },
                            {
                                "code": "CHART_DIRECTION_CONFLICT",
                                "message": "chart direction conflicts with lane side",
                                "severity": "WARN",
                            },
                        ],
                        "live_strategy_issues": [
                            {
                                "code": "STRATEGY_NOT_ELIGIBLE",
                                "message": "strategy profile is not live eligible",
                                "severity": "WARN",
                                "strategy_profile_evidence": {
                                    "profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                    "required_fix": "both live and pretrade feedback are negative",
                                    "live_net_jpy": -1501.57,
                                    "pretrade_net_jpy": -837.76,
                                },
                            }
                        ],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "range_trader:NZD_USD:SHORT:RANGE_ROTATION:MARKET",
                        "status": "DRY_RUN_PASSED",
                        "intent": {
                            "pair": "NZD_USD",
                            "side": "SHORT",
                            "order_type": "MARKET",
                            "market_context": {"method": "RANGE_ROTATION"},
                            "metadata": {"forecast_direction": "RANGE", "forecast_confidence": 0.63},
                        },
                        "risk_metrics": {"reward_risk": 2.85},
                        "risk_issues": [],
                        "live_strategy_issues": [
                            {
                                "code": "STRATEGY_NOT_ELIGIBLE",
                                "message": "strategy profile is not live eligible",
                                "severity": "WARN",
                                "strategy_profile_evidence": {
                                    "profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                    "required_fix": "both live and pretrade feedback are negative",
                                    "live_net_jpy": -1501.57,
                                    "pretrade_net_jpy": -837.76,
                                },
                            }
                        ],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "range_trader:NZD_USD:SHORT:RANGE_ROTATION:MARKET",
                        "status": "DRY_RUN_PASSED",
                        "intent": {
                            "pair": "NZD_USD",
                            "side": "SHORT",
                            "order_type": "MARKET",
                            "market_context": {"method": "RANGE_ROTATION"},
                            "metadata": {"forecast_direction": "RANGE", "forecast_confidence": 0.61},
                        },
                        "risk_metrics": {"reward_risk": 1.25},
                        "risk_issues": [],
                        "live_strategy_issues": [
                            {
                                "code": "STRATEGY_NOT_ELIGIBLE",
                                "message": "strategy profile is not live eligible",
                                "severity": "WARN",
                            }
                        ],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:STOP",
                        "status": "DRY_RUN_BLOCKED",
                        "risk_issues": [
                            {
                                "code": "REWARD_RISK_TOO_LOW",
                                "message": "reward/risk below floor",
                                "severity": "BLOCK",
                            }
                        ],
                        "live_strategy_issues": [],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "scalper:USD_JPY:LONG:EXECUTION:MARKET",
                        "status": "DRY_RUN_BLOCKED",
                        "risk_issues": [],
                        "live_strategy_issues": [],
                        "live_blockers": ["broker liquidity unavailable for order"],
                    },
                    {
                        "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION:MARKET",
                        "status": "LIVE_READY",
                        "risk_issues": [
                            {
                                "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                "message": "live-ready advisory must not be counted",
                                "severity": "WARN",
                            }
                        ],
                        "live_strategy_issues": [],
                        "live_blockers": [],
                    },
                ]
            }
        )

        all_families = {item["family"]: item for item in breakdown["all_non_live_ready"]}
        dry_run_families = {item["family"]: item for item in breakdown["dry_run_passed"]}
        self.assertEqual(all_families["forecast"]["lane_count"], 1)
        self.assertEqual(all_families["strategy_profile"]["lane_count"], 2)
        self.assertEqual(all_families["market_structure"]["lane_count"], 1)
        self.assertEqual(all_families["risk_geometry"]["lane_count"], 1)
        self.assertEqual(all_families["execution_liquidity"]["lane_count"], 1)
        self.assertNotIn("risk_geometry", dry_run_families)
        self.assertEqual(dry_run_families["strategy_profile"]["lane_count"], 2)
        self.assertEqual(
            breakdown["nearest_live_ready_candidates"][0]["lane_id"],
            "range_trader:NZD_USD:SHORT:RANGE_ROTATION:MARKET",
        )
        self.assertEqual(
            breakdown["nearest_live_ready_candidates"][0]["blocker_families"],
            ["strategy_profile"],
        )
        self.assertEqual(breakdown["nearest_live_ready_candidates"][0]["reward_risk"], 2.85)
        blocker_evidence = breakdown["nearest_live_ready_candidates"][0]["blockers"][0][
            "strategy_profile_evidence"
        ]
        self.assertEqual(blocker_evidence["profile_status"], "BLOCK_UNTIL_NEW_EVIDENCE")
        self.assertEqual(blocker_evidence["live_net_jpy"], -1501.57)
        candidate_lane_ids = [item["lane_id"] for item in breakdown["nearest_live_ready_candidates"]]
        self.assertEqual(len(candidate_lane_ids), len(set(candidate_lane_ids)))

    def test_blocks_missing_memory_projection_and_entry_thesis_holes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True, write_memory=False, projection_expired=True)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

            codes = {item["code"] for item in payload["findings"]}
            self.assertEqual(summary.status, STATUS_BLOCKED)
            self.assertGreaterEqual(summary.p0_findings, 3)
            self.assertEqual(payload["findings_count"], summary.findings)
            self.assertEqual(payload["p0_findings"], summary.p0_findings)
            self.assertEqual(payload["p1_findings"], summary.p1_findings)
            self.assertEqual(payload["p2_findings"], summary.p2_findings)
            self.assertIn("MEMORY_HEALTH_UNREADABLE", codes)
            self.assertIn("PROJECTION_LEDGER_EXPIRED_PENDING", codes)
            self.assertIn("ENTRY_THESIS_MISSING_FOR_OPEN_TRADES", codes)
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]
                finding_count = conn.execute("SELECT COUNT(*) FROM self_improvement_findings").fetchone()[0]
            self.assertEqual(run_count, 1)
            self.assertEqual(finding_count, summary.findings)

    def test_stale_memory_health_routes_to_refresh_before_old_blocker_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4, closed_pls=(100.0, 80.0, -50.0))
            files["memory"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": (_NOW - timedelta(minutes=5)).isoformat(),
                        "status": "MEMORY_HEALTH_BLOCKED",
                        "issues": [{"code": "SHORT_FORECAST_PAIR_STALE", "severity": "BLOCK"}],
                        "blockers": ["old forecast row predates old broker snapshot"],
                        "warnings": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("MEMORY_HEALTH_STALE", codes)
        self.assertNotIn("MEMORY_HEALTH_BLOCKED", codes)
        evidence = codes["MEMORY_HEALTH_STALE"]["evidence"]
        self.assertEqual(evidence["memory_health_generated_at_utc"], (_NOW - timedelta(minutes=5)).isoformat())
        self.assertEqual(evidence["stale_against"][0]["label"], "broker_snapshot")

    def test_learning_audit_quarantine_is_p1_not_global_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, 50.0),
            )
            files["learning"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "LEARNING_AUDIT_BLOCKED",
                        "blockers": [
                            "risk-increasing learning influence is active while recent effect window is negative"
                        ],
                        "warnings": [],
                        "checks": [
                            {
                                "check_name": "learning_influence_recent_outcome",
                                "status": "BLOCK",
                                "severity": "BLOCK",
                                "message": (
                                    "risk-increasing learning influence is active while recent effect "
                                    "window is negative"
                                ),
                            }
                        ],
                        "learning_influence": {
                            "influenced_lanes": 1,
                            "risk_increasing_lanes": 1,
                            "total_learning_score_delta": 8.0,
                            "lanes": [
                                {
                                    "lane_id": "trend_trader:AUD_USD:LONG:TREND_CONTINUATION",
                                    "learning_influences": ["ai_backtest_research_positive_edge"],
                                    "learning_score_delta": 8.0,
                                }
                            ],
                        },
                        "effect_metrics": {"closed_trades": 30, "net_jpy": -100.0, "profit_factor": 0.9},
                        "min_effect_sample": 3,
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertNotIn("LEARNING_AUDIT_BLOCKED", codes)
        self.assertEqual(codes["LEARNING_AUDIT_INFLUENCED_LANES_QUARANTINED"]["priority"], "P1")

    def test_memory_health_audited_snapshot_time_prevents_false_stale_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, closed_pls=(100.0, 80.0, -50.0))
            snapshot_ts = _NOW + timedelta(minutes=2)
            intents_ts = _NOW + timedelta(minutes=1)

            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["fetched_at_utc"] = snapshot_ts.isoformat()
            snapshot["account"]["fetched_at_utc"] = snapshot_ts.isoformat()
            files["snapshot"].write_text(json.dumps(snapshot))

            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = intents_ts.isoformat()
            files["intents"].write_text(json.dumps(intents))

            memory = json.loads(files["memory"].read_text())
            memory["generated_at_utc"] = _NOW.isoformat()
            memory["metrics"] = {
                "runtime": {
                    "snapshot_fetched_at_utc": snapshot_ts.isoformat(),
                    "order_intents_generated_at_utc": intents_ts.isoformat(),
                }
            }
            files["memory"].write_text(json.dumps(memory))

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("MEMORY_HEALTH_STALE", codes)

    def test_external_live_lock_defers_mid_refresh_memory_stale_judgment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4, closed_pls=(100.0, 80.0, -50.0))
            files["memory"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": (_NOW - timedelta(minutes=5)).isoformat(),
                        "status": "MEMORY_HEALTH_PASS",
                        "issues": [],
                        "blockers": [],
                        "warnings": [],
                    }
                )
            )
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
            (lock_dir / "command").write_text("cycle-refresh", encoding="utf-8")
            (lock_dir / "started_at_utc").write_text(
                (_NOW - timedelta(minutes=3)).isoformat(),
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir), "QR_AUTOTRADE_LOCK_HELD": ""},
                clear=False,
            ):
                summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("LIVE_RUNTIME_UPDATE_IN_PROGRESS", codes)
        self.assertNotIn("MEMORY_HEALTH_STALE", codes)
        self.assertNotIn("TARGET_OPEN_NO_LIVE_READY_LANES", codes)
        evidence = codes["LIVE_RUNTIME_UPDATE_IN_PROGRESS"]["evidence"]
        self.assertEqual(evidence["pid"], os.getpid())
        self.assertEqual(evidence["command"], "cycle-refresh")
        self.assertEqual(evidence["started_at_utc"], (_NOW - timedelta(minutes=3)).isoformat())
        self.assertGreaterEqual(evidence["lock_age_seconds"], 0.0)

    def test_external_live_lock_still_surfaces_coverage_perspective_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4, closed_pls=(100.0, 80.0, -50.0))
            files["coverage"].write_text(
                json.dumps(
                    {
                        "artifact_diagnostics": {},
                        "perspective_alignment_diagnostics": {
                            "status": "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED",
                            "pair_direction_groups": 2,
                            "range_forecast_method_mismatch_groups": 1,
                            "range_forecast_method_mismatch_lanes": 3,
                            "range_forecast_method_mismatch_top": [
                                {
                                    "pair": "AUD_JPY",
                                    "direction": "SHORT",
                                    "method_mismatch_lanes": 3,
                                    "range_rotation_lanes": 0,
                                    "range_rotation_live_ready_lanes": 0,
                                    "top_live_blocker_codes": [
                                        {"code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION", "count": 3}
                                    ],
                                }
                            ],
                        },
                    }
                )
            )
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir), "QR_AUTOTRADE_LOCK_HELD": ""},
                clear=False,
            ):
                summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("LIVE_RUNTIME_UPDATE_IN_PROGRESS", codes)
        self.assertNotIn("TARGET_OPEN_NO_LIVE_READY_LANES", codes)
        finding = codes["RANGE_FORECAST_METHOD_MISMATCH_REPAIR_REQUIRED"]
        self.assertEqual(finding["priority"], "P1")
        perspective = finding["evidence"]["perspective_alignment_diagnostics"]
        self.assertEqual(perspective["range_forecast_method_mismatch_lanes"], 3)
        self.assertEqual(perspective["range_forecast_method_mismatch_top"][0]["pair"], "AUD_JPY")

    def test_wrapper_owned_live_lock_still_allows_memory_stale_judgment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4, closed_pls=(100.0, 80.0, -50.0))
            files["memory"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": (_NOW - timedelta(minutes=5)).isoformat(),
                        "status": "MEMORY_HEALTH_PASS",
                        "issues": [],
                        "blockers": [],
                        "warnings": [],
                    }
                )
            )
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir), "QR_AUTOTRADE_LOCK_HELD": "1"},
                clear=False,
            ):
                summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertNotIn("LIVE_RUNTIME_UPDATE_IN_PROGRESS", codes)
        self.assertIn("MEMORY_HEALTH_STALE", codes)

    def test_missing_entry_thesis_ledger_without_open_trades_is_not_a_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["entry_thesis"].unlink()

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("ENTRY_THESIS_LEDGER_UNREADABLE", codes)
        self.assertEqual(summary.p0_findings, 0)

    def test_history_dedupes_identical_retry_inside_operational_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True, write_memory=False)

            first = _run(files, now=_NOW)
            second = _run(files, now=_NOW + timedelta(seconds=30))

            self.assertEqual(first.status, STATUS_BLOCKED)
            self.assertEqual(second.status, STATUS_BLOCKED)
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]
                finding_count = conn.execute("SELECT COUNT(*) FROM self_improvement_findings").fetchone()[0]
            self.assertEqual(run_count, 1)
            self.assertEqual(finding_count, first.findings)

    def test_history_dedupes_stale_gpt_retry_ignoring_streak_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )

            first = _run(files, now=_NOW)
            _run(files, now=_NOW + timedelta(seconds=30))

            self.assertEqual(first.status, STATUS_BLOCKED)
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]
                finding_count = conn.execute("SELECT COUNT(*) FROM self_improvement_findings").fetchone()[0]
                stale_streaks = [
                    json.loads(row[0]).get("current_streak")
                    for row in conn.execute(
                        """
                        SELECT evidence_json
                        FROM self_improvement_findings
                        WHERE code = 'LATEST_GPT_DECISION_STALE'
                        """
                    )
                ]
            self.assertEqual(run_count, 1)
            self.assertEqual(finding_count, first.findings)
            self.assertEqual(stale_streaks, [1])

    def test_pending_entry_lifecycle_flags_cancel_before_fill_churn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, 50.0),
            )
            _write_pending_cancel_churn_ledger(files["execution_db"])

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        lifecycle = payload["execution_quality"]["pending_entry_lifecycle"]
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("PENDING_ENTRY_FILL_RATE_WEAK", codes)
        self.assertEqual(codes["PENDING_ENTRY_FILL_RATE_WEAK"]["priority"], "P1")
        self.assertEqual(lifecycle["accepted_entry_orders"], 3)
        self.assertEqual(lifecycle["filled_entry_orders"], 0)
        self.assertEqual(lifecycle["canceled_before_fill_orders"], 3)
        self.assertEqual(lifecycle["canceled_before_fill_replaced_orders"], 0)
        self.assertEqual(lifecycle["canceled_before_fill_orphan_orders"], 3)
        self.assertEqual(lifecycle["cancel_before_fill_rate"], 1.0)
        self.assertEqual(lifecycle["cancel_replacement_rate"], 0.0)
        self.assertIn("Pending entry lifecycle", report)

    def test_pending_entry_lifecycle_flags_high_cancel_rate_even_with_fills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, 50.0),
            )
            _write_pending_mixed_cancel_churn_ledger(files["execution_db"])
            files["coverage"].write_text(
                json.dumps(
                    {
                        "status": "COVERAGE_OK",
                        "remaining_target_jpy": 1000.0,
                        "live_ready_reward_jpy": 1000.0,
                        "artifact_diagnostics": {},
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        lifecycle = payload["execution_quality"]["pending_entry_lifecycle"]
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("PENDING_ENTRY_CANCEL_RATE_HIGH", codes)
        self.assertEqual(codes["PENDING_ENTRY_CANCEL_RATE_HIGH"]["priority"], "P1")
        self.assertEqual(lifecycle["accepted_entry_orders"], 5)
        self.assertEqual(lifecycle["filled_entry_orders"], 2)
        self.assertEqual(lifecycle["canceled_before_fill_orders"], 3)
        self.assertEqual(lifecycle["canceled_before_fill_replaced_orders"], 0)
        self.assertEqual(lifecycle["canceled_before_fill_orphan_orders"], 3)
        self.assertAlmostEqual(lifecycle["cancel_before_fill_rate"], 0.6)
        self.assertEqual(lifecycle["cancel_replacement_rate"], 0.0)
        self.assertEqual(payload["root_cause_focus"]["primary"]["family"], "EXECUTION_LIFECYCLE")

    def test_pending_entry_lifecycle_distinguishes_replaced_from_orphan_cancels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, 50.0),
            )
            _write_pending_replacement_churn_ledger(files["execution_db"])

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        lifecycle = payload["execution_quality"]["pending_entry_lifecycle"]
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("PENDING_ENTRY_FILL_RATE_WEAK", codes)
        self.assertEqual(lifecycle["accepted_entry_orders"], 4)
        self.assertEqual(lifecycle["canceled_before_fill_orders"], 3)
        self.assertEqual(lifecycle["canceled_before_fill_replaced_orders"], 1)
        self.assertEqual(lifecycle["canceled_before_fill_orphan_orders"], 2)
        self.assertAlmostEqual(lifecycle["cancel_replacement_rate"], 1 / 3)
        samples = codes["PENDING_ENTRY_FILL_RATE_WEAK"]["evidence"]["samples"]
        replaced = next(item for item in samples if item["order_id"] == "replace-source")
        self.assertEqual(replaced["replaced_with_order_id"], "replace-next")
        self.assertAlmostEqual(replaced["replacement_after_min"], 5.0)
        groups = lifecycle["canceled_before_fill_orphan_groups"]
        self.assertEqual(groups[0]["pair"], "CAD_CHF")
        self.assertEqual(groups[0]["side"], "LONG")
        self.assertEqual(groups[0]["method"], "RANGE_ROTATION")
        self.assertEqual(groups[0]["count"], 1)

    def test_current_pending_reconcile_flags_cap_candidate_and_advice_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, 50.0),
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["home_conversions"] = {"USD": 150.0}
            snapshot["orders"] = [
                {
                    "order_id": "risk-pending",
                    "pair": "EUR_USD",
                    "order_type": "STOP",
                    "state": "PENDING",
                    "units": 2000,
                    "price": 1.171,
                    "owner": "trader",
                    "trade_id": None,
                    "raw": {
                        "createTime": (_NOW - timedelta(minutes=40)).isoformat(),
                        "clientExtensions": {
                            "comment": (
                                "qr-vnext lane=trend_trader:EUR_USD:LONG:TREND_CONTINUATION "
                                "desk=trend_trader"
                            ),
                            "tag": "trader",
                        },
                        "stopLossOnFill": {"price": "1.1600"},
                    },
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))
            files["target"].write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1000.0,
                        "per_trade_risk_budget_jpy": 100.0,
                    }
                )
            )
            files["intents"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "results": [
                            {
                                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                "status": "DRY_RUN_BLOCKED",
                                "intent": {
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "order_type": "STOP-ENTRY",
                                },
                                "risk_issues": [
                                    {
                                        "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                        "severity": "BLOCK",
                                    }
                                ],
                                "live_blockers": ["forecast confidence below entry grade"],
                            }
                        ],
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "ATTACK_ADVICE_READY",
                        "recommended_now_lane_ids": ["trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("PENDING_ENTRY_CANCEL_REVIEW_REQUIRED", codes)
        review = payload["execution_quality"]["pending_entry_reconcile"]
        self.assertEqual(review["cancel_review_order_ids"], ["risk-pending"])
        order_review = review["orders"][0]
        reason_codes = {item["code"] for item in order_review["review_reasons"]}
        self.assertIn("PENDING_ATTACHED_SL_RISK_EXCEEDS_CAP", reason_codes)
        self.assertIn("PENDING_CURRENT_CANDIDATE_NOT_LIVE_READY", reason_codes)
        self.assertIn("PENDING_ATTACK_ADVICE_NOT_CURRENT", reason_codes)
        self.assertGreater(order_review["attached_sl_risk_jpy"], 100.0)
        self.assertIn("Pending entry reconcile", report)

    def test_repeated_self_improvement_finding_surfaces_anti_loop_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, 50.0),
            )

            _run(files, now=_NOW)
            _run(files, now=_NOW + timedelta(minutes=3))
            third = _run(files, now=_NOW + timedelta(minutes=6))
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(third.status, STATUS_BLOCKED)
        self.assertIn("REPEATED_SELF_IMPROVEMENT_LOOP", codes)
        loop = codes["REPEATED_SELF_IMPROVEMENT_LOOP"]
        self.assertEqual(loop["priority"], "P1")
        self.assertEqual(loop["evidence"]["current_streak"], 3)
        self.assertEqual(loop["evidence"]["previous_streak"], 2)
        self.assertEqual(loop["evidence"]["repeated_code"], "TARGET_OPEN_NO_LIVE_READY_LANES")

    def test_root_cause_focus_prioritizes_forecast_adverse_path_over_process_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, 50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "resolution_evidence": "2026-06-16T04:44:00Z invalidation 1.16800 touched before target 1.17200",
                        "cycle_id": f"invalidation-cycle-{idx}",
                    }
                )
            for idx in range(2):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 14)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT",
                        "resolution_evidence": "2026-06-16T04:24:00Z target 1.17200 touched before invalidation 1.16800",
                        "cycle_id": f"hit-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            _run(files, now=_NOW)
            _run(files, now=_NOW + timedelta(minutes=3))
            _run(files, now=_NOW + timedelta(minutes=6))
            payload = json.loads(files["output"].read_text())

        root_focus = payload["root_cause_focus"]
        self.assertEqual(root_focus["status"], "FOCUSED")
        self.assertEqual(root_focus["primary"]["family"], "FORECAST_ADVERSE_PATH")
        self.assertIn(
            "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
            root_focus["primary"]["supporting_codes"],
        )
        self.assertEqual(payload["next_actions"][0]["code"], "ROOT_CAUSE_FOCUS:FORECAST_ADVERSE_PATH")
        self.assertIn("REPEATED_SELF_IMPROVEMENT_LOOP", {item["code"] for item in payload["findings"]})

    def test_action_required_for_hidden_open_loss_and_low_market_rr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=0.8,
                unrealized_pl_jpy=-300.0,
                closed_pls=(100.0, 80.0, -50.0),
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("OPEN_LOSS_EXCEEDS_24H_REALIZED_GAIN", codes)
        self.assertIn("LIVE_READY_MARKET_RR_BELOW_ONE", codes)
        self.assertEqual(summary.live_ready_lanes, 1)

    def test_directional_forecast_timeout_only_samples_are_p1_learning_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["projection_ledger"].write_text(
                "\n".join(
                    json.dumps(
                        {
                            "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "signal_name": "directional_forecast",
                            "predicted_target_price": 1.1720,
                            "predicted_invalidation_price": 1.1680,
                            "resolution_window_min": 60.0,
                            "resolution_status": "TIMEOUT",
                            "resolution_evidence": "no candle truth for projection window",
                            "cycle_id": f"cycle-{idx}",
                        }
                    )
                    for idx in range(10)
                )
                + "\n"
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("DIRECTIONAL_FORECAST_CALIBRATION_UNRESOLVED", codes)
        self.assertEqual(codes["DIRECTIONAL_FORECAST_CALIBRATION_UNRESOLVED"]["priority"], "P1")
        self.assertEqual(codes["DIRECTIONAL_FORECAST_CALIBRATION_UNRESOLVED"]["evidence"]["status_counts"]["TIMEOUT"], 10)

    def test_directional_forecast_low_hit_rate_is_p1_forecast_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx == 0 else "MISS",
                        "cycle_id": f"cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("DIRECTIONAL_FORECAST_HIT_RATE_WEAK", codes)
        evidence = codes["DIRECTIONAL_FORECAST_HIT_RATE_WEAK"]["evidence"]
        self.assertEqual(evidence["samples"], 10)
        self.assertAlmostEqual(evidence["hit_rate"], 0.1)
        self.assertTrue(evidence["worst_buckets"])

    def test_directional_forecast_watch_only_samples_do_not_trigger_entry_grade_hit_rate_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(12):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.12,
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "resolution_evidence": (
                            "2026-06-16T04:44:00Z invalidation 1.16800 touched before target 1.17200"
                        ),
                        "cycle_id": f"watch-only-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("DIRECTIONAL_FORECAST_ENTRY_GRADE_SAMPLE_SHORTFALL", codes)
        self.assertNotIn("DIRECTIONAL_FORECAST_HIT_RATE_WEAK", codes)
        self.assertNotIn("DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT", codes)
        finding = codes["DIRECTIONAL_FORECAST_ENTRY_GRADE_SAMPLE_SHORTFALL"]
        self.assertEqual(finding["priority"], "P1")
        evidence = finding["evidence"]
        self.assertEqual(evidence["entry_grade_samples"], 0)
        self.assertEqual(evidence["watch_only_movement_samples"], 12)
        self.assertEqual(evidence["movement_samples"], 12)

    def test_directional_forecast_low_hit_rate_excludes_range_box_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "cycle_id": f"movement-cycle-{idx}",
                    }
                )
            for idx in range(20):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "RANGE",
                        "regime_at_emission": "RANGE",
                        "signal_name": "directional_forecast",
                        "predicted_range_low_price": 1.1680,
                        "predicted_range_high_price": 1.1720,
                        "resolution_window_min": 120.0,
                        "resolution_status": "HIT",
                        "cycle_id": f"range-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        hit_rate = codes["DIRECTIONAL_FORECAST_HIT_RATE_WEAK"]
        evidence = hit_rate["evidence"]
        self.assertEqual(evidence["samples"], 10)
        self.assertEqual(evidence["hit_count"], 0)
        self.assertAlmostEqual(evidence["hit_rate"], 0.0)
        self.assertEqual(evidence["range_samples_excluded"], 20)
        self.assertEqual(evidence["total_calibrated_samples"], 30)

    def test_directional_forecast_no_touch_misses_are_target_timeout_not_hit_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(15):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "resolution_evidence": "target 1.17200 not reached before invalidation 1.16800",
                        "cycle_id": f"target-timeout-cycle-{idx}",
                    }
                )
            for idx in range(10):
                status = "HIT" if idx < 5 else "MISS"
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 20)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": status,
                        "resolution_evidence": (
                            "2026-06-16T04:24:00Z target 1.17200 touched before invalidation 1.16800"
                            if status == "HIT"
                            else "2026-06-16T04:44:00Z invalidation 1.16800 touched before target 1.17200"
                        ),
                        "cycle_id": f"touch-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("DIRECTIONAL_FORECAST_TARGET_TIMEOUT_DOMINANT", codes)
        self.assertNotIn("DIRECTIONAL_FORECAST_HIT_RATE_WEAK", codes)
        evidence = codes["DIRECTIONAL_FORECAST_TARGET_TIMEOUT_DOMINANT"]["evidence"]
        self.assertEqual(evidence["target_timeout_samples"], 15)
        self.assertEqual(evidence["touch_calibrated_samples"], 10)
        self.assertAlmostEqual(evidence["target_timeout_rate"], 0.6)

    def test_directional_forecast_invalidation_first_dominant_is_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "resolution_evidence": "2026-06-16T04:44:00Z invalidation 1.16800 touched before target 1.17200",
                        "cycle_id": f"invalidation-cycle-{idx}",
                    }
                )
            for idx in range(2):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 14)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT",
                        "resolution_evidence": "2026-06-16T04:24:00Z target 1.17200 touched before invalidation 1.16800",
                        "cycle_id": f"hit-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        finding = codes["DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT"]
        self.assertEqual(finding["priority"], "P1")
        evidence = finding["evidence"]
        self.assertEqual(evidence["samples"], 12)
        self.assertEqual(evidence["invalidation_first_count"], 10)
        self.assertAlmostEqual(evidence["invalidation_first_rate"], 0.8333)
        self.assertEqual(evidence["worst_buckets"][0]["pair"], "EUR_USD")

    def test_directional_forecast_historical_weakness_is_p2_when_recent_window_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(days=8, hours=idx)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "cycle_id": f"old-cycle-{idx}",
                    }
                )
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx < 6 else "MISS",
                        "cycle_id": f"recent-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        hit_rate = codes["DIRECTIONAL_FORECAST_HIT_RATE_WEAK"]
        self.assertEqual(hit_rate["priority"], "P2")
        self.assertTrue(hit_rate["evidence"]["recent_recovered"])
        self.assertAlmostEqual(hit_rate["evidence"]["window_hit_rates"]["7d"]["hit_rate"], 0.6)
        bucket = codes["DIRECTIONAL_FORECAST_BUCKET_HIT_RATE_WEAK"]
        self.assertEqual(bucket["priority"], "P2")
        self.assertTrue(bucket["evidence"]["recent_recovered"])

    def test_directional_forecast_timeout_dominant_is_p1_calibration_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(2):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx == 0 else "MISS",
                        "cycle_id": f"cycle-hitmiss-{idx}",
                    }
                )
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 4)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "TIMEOUT",
                        "cycle_id": f"cycle-timeout-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("DIRECTIONAL_FORECAST_CALIBRATION_TIMEOUT_DOMINANT", codes)
        evidence = codes["DIRECTIONAL_FORECAST_CALIBRATION_TIMEOUT_DOMINANT"]["evidence"]
        self.assertEqual(evidence["calibrated_samples"], 2)
        self.assertEqual(evidence["status_counts"]["TIMEOUT"], 10)
        self.assertLess(evidence["calibration_coverage"], evidence["min_coverage"])

    def test_directional_forecast_missing_geometry_is_p1_calibration_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(2):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx == 0 else "MISS",
                        "cycle_id": f"cycle-hitmiss-{idx}",
                    }
                )
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 4)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": None,
                        "predicted_invalidation_price": None,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx < 6 else "MISS",
                        "cycle_id": f"cycle-legacy-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("DIRECTIONAL_FORECAST_CALIBRATION_GEOMETRY_MISSING", codes)
        self.assertNotIn("DIRECTIONAL_FORECAST_CALIBRATION_TIMEOUT_DOMINANT", codes)
        evidence = codes["DIRECTIONAL_FORECAST_CALIBRATION_GEOMETRY_MISSING"]["evidence"]
        self.assertEqual(evidence["calibrated_samples"], 2)
        self.assertEqual(evidence["missing_geometry_samples"], 10)
        self.assertLess(evidence["calibration_coverage"], evidence["min_coverage"])

    def test_directional_forecast_old_missing_geometry_is_p2_when_recent_geometry_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(14):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(days=8, hours=idx)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": None,
                        "predicted_invalidation_price": None,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "cycle_id": f"old-legacy-{idx}",
                    }
                )
            for idx in range(12):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx < 7 else "MISS",
                        "cycle_id": f"recent-geometry-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        finding = codes["DIRECTIONAL_FORECAST_CALIBRATION_GEOMETRY_MISSING"]
        self.assertEqual(finding["priority"], "P2")
        evidence = finding["evidence"]
        self.assertTrue(evidence["recent_recovered"])
        self.assertEqual(evidence["recent_24h_rows"], 12)
        self.assertEqual(evidence["recent_24h_calibrated_samples"], 12)
        self.assertGreaterEqual(evidence["recent_24h_calibration_coverage"], evidence["min_coverage"])

    def test_order_intents_without_market_context_refs_are_p1_when_matrix_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["market_context_matrix"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "pairs": {"EUR_USD": {"LONG": {"support": []}}},
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_MISSING", codes)
        finding = codes["ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_MISSING"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["candidate_count"], 1)
        self.assertEqual(finding["evidence"]["with_context_refs"], 0)

    def test_forecast_history_duplicate_cycle_pair_is_p1_measurement_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["forecast_history"].write_text(
                "\n".join(
                    json.dumps(
                        {
                            "timestamp_utc": (_NOW + timedelta(seconds=idx)).isoformat(),
                            "cycle_id": "cycle-dup",
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "confidence": 0.62,
                        }
                    )
                    for idx in range(2)
                )
                + "\n"
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("FORECAST_HISTORY_DUPLICATE_CYCLE_PAIR", codes)
        finding = codes["FORECAST_HISTORY_DUPLICATE_CYCLE_PAIR"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["duplicate_cycle_pair_groups"], 1)
        self.assertEqual(finding["evidence"]["examples"][0]["pair"], "EUR_USD")
        self.assertEqual(finding["evidence"]["examples"][0]["count"], 2)

    def test_legacy_forecast_history_phantom_clusters_are_p2_dedupe_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["forecast_history"].write_text(
                "\n".join(
                    json.dumps(
                        {
                            "timestamp_utc": _NOW.isoformat(),
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "confidence": 0.62,
                        }
                    )
                    for _idx in range(3)
                )
                + "\n"
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS", codes)
        finding = codes["FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS"]
        self.assertEqual(finding["priority"], "P2")
        self.assertEqual(finding["evidence"]["phantom_clusters"], 1)

    def test_market_context_ref_on_intent_satisfies_context_evidence_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["market_context_matrix"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "pairs": {"EUR_USD": {"LONG": {"support": []}}},
                    }
                )
            )
            intents = json.loads(files["intents"].read_text())
            intents["results"][0]["intent"]["metadata"]["market_context_matrix_ref"] = "matrix:EUR_USD:LONG"
            files["intents"].write_text(json.dumps(intents))

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_MISSING", codes)

    def test_order_intents_predating_matrix_with_live_ready_lane_is_p0_stale_context_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["market_context_matrix"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "pairs": {"EUR_USD": {"LONG": {"support": []}}},
                    }
                )
            )
            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = (_NOW - timedelta(hours=1)).isoformat()
            intents["results"][0]["intent"]["metadata"]["market_context_matrix_ref"] = "matrix:EUR_USD:LONG"
            files["intents"].write_text(json.dumps(intents))

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(summary.p0_findings, 1)
        self.assertIn("ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_STALE", codes)
        self.assertNotIn("ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_MISSING", codes)
        finding = codes["ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_STALE"]
        self.assertEqual(finding["priority"], "P0")
        self.assertEqual(finding["evidence"]["candidate_count"], 1)
        self.assertEqual(finding["evidence"]["live_ready_lanes"], 1)
        self.assertEqual(finding["evidence"]["with_context_refs"], 1)

    def test_order_intents_predating_matrix_without_live_ready_lane_stays_p1_context_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
                pending_entry=True,
            )
            files["market_context_matrix"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "pairs": {"EUR_USD": {"LONG": {"support": []}}},
                    }
                )
            )
            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = (_NOW - timedelta(hours=1)).isoformat()
            intents["results"][0]["status"] = "DRY_RUN_BLOCKED"
            intents["results"][0]["intent"]["metadata"]["market_context_matrix_ref"] = "matrix:EUR_USD:LONG"
            files["intents"].write_text(json.dumps(intents))

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_STALE", codes)
        finding = codes["ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_STALE"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["live_ready_lanes"], 0)

    def test_unattributable_close_gate_ablation_is_p1_assumption_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "close_events": 8,
                                "bot_attributed_close_events": 0,
                                "gateway_close_sent_events": 0,
                                "loss_side_market_close_count": 5,
                                "loss_side_market_close_net_jpy": -1200.0,
                                "broker_accepted_without_gateway_loss_side_market_close_count": 2,
                                "broker_accepted_without_gateway_loss_side_market_close_source_counts": {
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 2
                                },
                                "broker_accepted_without_gateway_loss_side_market_close_evidence_counts": {
                                    "NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 2,
                                    "NO_CLIENT_EXTENSION": 2,
                                },
                                "unattributed_loss_side_market_close_count": 5,
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("CLOSE_GATE_ABLATION_NOT_ATTRIBUTABLE", codes)
        finding = codes["CLOSE_GATE_ABLATION_NOT_ATTRIBUTABLE"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["gateway_close_sent_events"], 0)
        self.assertIn("direct/manual", finding["next_action"])
        self.assertEqual(
            finding["evidence"]["broker_accepted_without_gateway_loss_side_market_close_source_counts"],
            {"DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 2},
        )
        self.assertEqual(
            finding["evidence"]["broker_accepted_without_gateway_loss_side_market_close_evidence_counts"],
            {"NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 2, "NO_CLIENT_EXTENSION": 2},
        )

    def test_close_gate_ablation_with_trader_entry_source_requests_receipt_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "close_events": 8,
                                "bot_attributed_close_events": 8,
                                "gateway_close_sent_events": 0,
                                "loss_side_market_close_count": 5,
                                "loss_side_market_close_net_jpy": -1200.0,
                                "broker_accepted_without_gateway_loss_side_market_close_count": 3,
                                "broker_accepted_without_gateway_loss_side_market_close_source_counts": {
                                    "TRADER_ENTRY_LANE_ID": 2,
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 1,
                                },
                                "broker_accepted_without_gateway_loss_side_market_close_evidence_counts": {
                                    "NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 3,
                                    "TRADER_ENTRY_LANE_ID": 2,
                                    "NO_CLIENT_EXTENSION": 1,
                                },
                                "unattributed_loss_side_market_close_count": 0,
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        finding = codes["CLOSE_GATE_ABLATION_NOT_ATTRIBUTABLE"]
        self.assertIn("trader-owned entries", finding["next_action"])
        self.assertIn("GATEWAY_TRADE_CLOSE_SENT", finding["next_action"])
        self.assertIn("1 residual direct/manual close", finding["next_action"])

    def test_external_manual_close_residual_does_not_keep_close_gate_ablation_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "close_events": 8,
                                "bot_attributed_close_events": 8,
                                "gateway_close_sent_events": 4,
                                "broker_trade_close_accept_events": 4,
                                "loss_side_market_close_count": 5,
                                "loss_side_market_close_net_jpy": -1200.0,
                                "broker_accepted_without_gateway_loss_side_market_close_count": 1,
                                "broker_accepted_without_gateway_loss_side_market_close_source_counts": {
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 1,
                                },
                                "broker_accepted_without_gateway_loss_side_market_close_evidence_counts": {
                                    "NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 1,
                                    "NO_CLIENT_EXTENSION": 1,
                                },
                                "broker_accepted_without_gateway_policy_gap_loss_side_market_close_count": 0,
                                "broker_accepted_without_gateway_policy_gap_loss_side_market_close_source_counts": {},
                                "broker_accepted_without_gateway_policy_gap_loss_side_market_close_evidence_counts": {},
                                "broker_accepted_without_gateway_external_loss_side_market_close_count": 1,
                                "broker_accepted_without_gateway_external_loss_side_market_close_source_counts": {
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 1,
                                },
                                "unattributed_loss_side_market_close_count": 0,
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("CLOSE_GATE_ABLATION_NOT_ATTRIBUTABLE", codes)

    def test_legacy_review_exit_close_ablation_remains_p1_assumption_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "close_events": 8,
                                "bot_attributed_close_events": 8,
                                "gateway_close_sent_events": 3,
                                "broker_trade_close_accept_events": 3,
                                "loss_side_market_close_count": 3,
                                "loss_side_market_close_net_jpy": -900.0,
                                "gateway_gpt_close_loss_side_market_close_count": 0,
                                "gateway_review_exit_loss_side_market_close_count": 3,
                                "gateway_review_exit_loss_side_market_close_net_jpy": -900.0,
                                "broker_accepted_without_gateway_loss_side_market_close_count": 0,
                                "unattributed_loss_side_market_close_count": 0,
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("LEGACY_REVIEW_EXIT_CLOSE_DRAG", codes)
        finding = codes["LEGACY_REVIEW_EXIT_CLOSE_DRAG"]
        self.assertIn("legacy REVIEW_EXIT", finding["next_action"])
        self.assertEqual(finding["evidence"]["gateway_review_exit_loss_side_market_close_count"], 3)

    def test_historical_review_exit_close_drag_is_p2_when_no_24h_losses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "close_events": 8,
                                "bot_attributed_close_events": 8,
                                "gateway_close_sent_events": 3,
                                "broker_trade_close_accept_events": 3,
                                "loss_side_market_close_count": 3,
                                "loss_side_market_close_net_jpy": -900.0,
                                "gateway_gpt_close_loss_side_market_close_count": 0,
                                "gateway_review_exit_loss_side_market_close_count": 3,
                                "gateway_review_exit_loss_side_market_close_net_jpy": -900.0,
                                "gateway_review_exit_recent_24h_loss_side_market_close_count": 0,
                                "gateway_review_exit_recent_24h_loss_side_market_close_net_jpy": 0.0,
                                "gateway_review_exit_recent_7d_loss_side_market_close_count": 1,
                                "gateway_review_exit_recent_7d_loss_side_market_close_net_jpy": -172.0,
                                "gateway_review_exit_latest_loss_side_market_close_ts_utc": (
                                    "2026-05-14T14:44:38+00:00"
                                ),
                                "broker_accepted_without_gateway_loss_side_market_close_count": 0,
                                "unattributed_loss_side_market_close_count": 0,
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("LEGACY_REVIEW_EXIT_CLOSE_DRAG", codes)
        finding = codes["LEGACY_REVIEW_EXIT_HISTORICAL_DRAG"]
        self.assertEqual(finding["priority"], "P2")
        self.assertEqual(finding["evidence"]["gateway_review_exit_recent_24h_loss_side_market_close_count"], 0)
        self.assertEqual(finding["evidence"]["gateway_review_exit_recent_7d_loss_side_market_close_count"], 1)

    def test_profitable_backtest_edges_missing_from_coverage_are_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "artifact_diagnostics": {
                            "profitable_bucket_coverage": {
                                "source_status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                                "live_permission": False,
                                "positive_pair_directions": 2,
                                "positive_managed_net_jpy": 1500.0,
                                "state_counts": {
                                    "NO_CURRENT_LANE": 1,
                                    "SPREAD_NORMALIZED_LIVE_BLOCKED": 1,
                                },
                                "blocked_or_missing_top": [
                                    {
                                        "pair": "USD_CAD",
                                        "direction": "LONG",
                                        "coverage_state": "NO_CURRENT_LANE",
                                        "managed_net_jpy": 900.0,
                                        "raw_net_jpy": 800.0,
                                        "trades": 10,
                                        "days": 3,
                                        "current_lane_count": 0,
                                        "spread_normalized_candidate_count": 0,
                                        "spread_normalized_no_live_blocker_count": 0,
                                        "top_blockers": [],
                                        "strategy_profile_status": "MINE_MISSED_EDGE",
                                        "strategy_profile_required_fix": "candidate not surfaced",
                                        "strategy_profile_blocks_live": True,
                                        "matrix_support_count": 8,
                                        "matrix_reject_count": 1,
                                        "matrix_warning_count": 2,
                                        "matrix_strongest_support": "USD_CAD DXY and oil context align LONG",
                                        "matrix_strongest_reject": "spread stressed",
                                        "matrix_cross_asset_context": [
                                            "GOLD_CONTEXT_TECHNICAL_DIRECTION: XAU_USD maps to LONG",
                                            "OIL_CONTEXT_TECHNICAL_DIRECTION: WTICO_USD maps to LONG",
                                        ],
                                        "matrix_support_context": [
                                            "GOLD_CONTEXT_TECHNICAL_DIRECTION: XAU_USD maps to LONG",
                                            "OIL_CONTEXT_TECHNICAL_DIRECTION: WTICO_USD maps to LONG",
                                        ],
                                        "matrix_reject_context": [],
                                        "same_side_matrix_context_supported": True,
                                    },
                                    {
                                        "pair": "EUR_USD",
                                        "direction": "SHORT",
                                        "coverage_state": "SPREAD_NORMALIZED_NO_LIVE_BLOCKER",
                                        "managed_net_jpy": 600.0,
                                        "current_lane_count": 1,
                                    },
                                ],
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP", codes)
        self.assertIn("MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE", codes)
        gap = codes["PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP"]
        self.assertEqual(gap["priority"], "P1")
        self.assertEqual(gap["evidence"]["blocked_edges"][0]["pair"], "USD_CAD")
        supported = codes["MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE"]
        self.assertIn("GOLD_CONTEXT_TECHNICAL_DIRECTION", supported["evidence"]["supported_edges"][0]["matrix_cross_asset_context"][0])

    def test_forecast_gated_profitable_edges_do_not_become_coverage_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "artifact_diagnostics": {
                            "profitable_bucket_coverage": {
                                "source_status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                                "live_permission": False,
                                "positive_pair_directions": 1,
                                "positive_managed_net_jpy": 900.0,
                                "state_counts": {"SURFACED_BUT_BLOCKED": 1},
                                "blocked_or_missing_top": [
                                    {
                                        "pair": "EUR_USD",
                                        "direction": "LONG",
                                        "coverage_state": "SURFACED_BUT_BLOCKED",
                                        "managed_net_jpy": 900.0,
                                        "raw_net_jpy": 800.0,
                                        "trades": 10,
                                        "days": 3,
                                        "current_lane_count": 4,
                                        "spread_normalized_candidate_count": 0,
                                        "spread_normalized_no_live_blocker_count": 0,
                                        "top_blockers": [
                                            "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                            "EUR_USD LONG forecast RANGE confidence 0.38 < 0.55",
                                        ],
                                        "strategy_profile_status": "CANDIDATE",
                                        "strategy_profile_required_fix": "eligible but forecast blocked",
                                        "strategy_profile_blocks_live": False,
                                        "matrix_support_count": 8,
                                        "matrix_reject_count": 1,
                                        "matrix_support_context": [
                                            "GOLD_CONTEXT_TECHNICAL_DIRECTION: XAU_USD maps to LONG",
                                        ],
                                        "same_side_matrix_context_supported": True,
                                    }
                                ],
                            }
                        }
                    }
                )
            )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertNotIn("PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP", codes)
        self.assertNotIn("MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE", codes)
        finding = codes["PROFITABLE_BACKTEST_EDGE_FORECAST_GATED"]
        self.assertEqual(finding["priority"], "P2")
        self.assertEqual(finding["evidence"]["forecast_gated_edges"][0]["pair"], "EUR_USD")

    def test_strategy_gated_profitable_edges_do_not_become_coverage_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "artifact_diagnostics": {
                            "profitable_bucket_coverage": {
                                "source_status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                                "live_permission": False,
                                "positive_pair_directions": 1,
                                "positive_managed_net_jpy": 700.0,
                                "state_counts": {"NO_CURRENT_LANE": 1},
                                "blocked_or_missing_top": [
                                    {
                                        "pair": "USD_JPY",
                                        "direction": "LONG",
                                        "coverage_state": "NO_CURRENT_LANE",
                                        "managed_net_jpy": 700.0,
                                        "raw_net_jpy": 650.0,
                                        "trades": 7,
                                        "days": 2,
                                        "current_lane_count": 0,
                                        "spread_normalized_candidate_count": 0,
                                        "spread_normalized_no_live_blocker_count": 0,
                                        "top_blockers": [],
                                        "strategy_profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                        "strategy_profile_required_fix": "current evidence required",
                                        "strategy_profile_blocks_live": True,
                                        "matrix_support_count": 6,
                                        "matrix_reject_count": 0,
                                        "matrix_support_context": [
                                            "DXY_CONTEXT_TECHNICAL_DIRECTION: DXY maps to LONG",
                                        ],
                                        "same_side_matrix_context_supported": True,
                                    }
                                ],
                            }
                        }
                    }
                )
            )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertNotIn("PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP", codes)
        self.assertNotIn("MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE", codes)
        finding = codes["PROFITABLE_BACKTEST_EDGE_STRATEGY_GATED"]
        self.assertEqual(finding["priority"], "P2")
        self.assertEqual(finding["evidence"]["strategy_gated_edges"][0]["pair"], "USD_JPY")

    def test_lane_only_verification_blockers_do_not_mask_opportunity_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, verification_lane_blockers=True)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(summary.p0_findings, 1)
        self.assertIn("TARGET_OPEN_NO_LIVE_READY_LANES", codes)
        self.assertIn("VERIFICATION_LEDGER_LANE_BLOCKERS_RECORDED", codes)
        self.assertNotIn("VERIFICATION_LEDGER_BLOCKED", codes)

    def test_pending_entry_downgrades_no_live_ready_hole_from_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, pending_entry=True)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertEqual(codes["TARGET_OPEN_NO_LIVE_READY_LANES"]["priority"], "P1")
        self.assertEqual(
            codes["TARGET_OPEN_NO_LIVE_READY_LANES"]["evidence"]["trader_pending_entry_orders"][0]["order_id"],
            "P1",
        )
        self.assertEqual(payload["runtime"]["open_trader_pending_entries"], 1)

    def test_market_evidence_refresh_downgrades_no_live_ready_hole_from_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["coverage"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "COVERAGE_GAP",
                        "artifact_diagnostics": {
                            "requires_market_evidence_refresh": True,
                            "all_lanes_spread_blocked": True,
                            "all_lanes_quote_stale": False,
                            "quote_stale_result_count": 8,
                            "spread_normalized_candidate_count": 2,
                            "spread_normalized_candidate_reward_jpy": 2376.0,
                        },
                        "action_items": [
                            "refresh broker-snapshot and generate-intents after quotes and spreads are tradable",
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        finding = codes["TARGET_OPEN_NO_LIVE_READY_LANES"]
        evidence = finding["evidence"]["coverage_market_evidence_refresh"]
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertEqual(finding["priority"], "P1")
        self.assertTrue(evidence["requires_market_evidence_refresh"])
        self.assertTrue(evidence["all_lanes_spread_blocked"])
        self.assertEqual(evidence["spread_normalized_candidate_count"], 2)

    def test_no_live_ready_evidence_names_dry_run_passed_live_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT",
                                "status": "DRY_RUN_PASSED",
                                "intent": {
                                    "side": "SHORT",
                                    "order_type": "LIMIT",
                                    "metadata": {
                                        "opportunity_mode": "HARVEST",
                                        "opportunity_mode_reason": "tp_target_intent=HARVEST",
                                        "opportunity_mode_reward_risk": 1.18,
                                        "tp_target_intent": "HARVEST",
                                        "forecast_direction": "DOWN",
                                        "forecast_confidence": 0.311,
                                        "forecast_raw_confidence": 0.5244919527711897,
                                        "chart_direction_bias": "LONG",
                                        "forecast_market_support": {
                                            "aligned_projection_count": 1,
                                            "best_hit_rate": 0.82,
                                            "best_samples": 100,
                                            "direction": "DOWN",
                                            "ok": True,
                                            "reason": (
                                                "liquidity_sweep_high DOWN hit_rate=0.82 "
                                                "samples=100 supports weak calibrated forecast"
                                            ),
                                            "signals": [
                                                {
                                                    "confidence": 0.9922,
                                                    "direction": "DOWN",
                                                    "hit_rate": 0.82,
                                                    "name": "liquidity_sweep_high",
                                                    "samples": 100,
                                                    "timeframe": "M15",
                                                }
                                            ],
                                            "timing_projection_count": 0,
                                            "unselected_reason": "",
                                        },
                                    },
                                },
                                "risk_issues": [
                                    {
                                        "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                        "message": "forecast confidence below live floor",
                                        "severity": "WARN",
                                    }
                                ],
                                "strategy_issues": [
                                    {
                                        "code": "STRATEGY_PROFILE_MISSING",
                                        "message": "dry-run advisory profile warning",
                                        "severity": "WARN",
                                    }
                                ],
                                "live_strategy_issues": [
                                    {
                                        "code": "STRATEGY_NOT_ELIGIBLE",
                                        "message": "strategy profile is not live eligible",
                                        "severity": "WARN",
                                    }
                                ],
                                "live_blockers": [],
                                "risk_metrics": {
                                    "reward_jpy": 420.0,
                                    "reward_risk": 1.18,
                                },
                            }
                        ]
                    }
                )
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "opportunity_modes": {
                            "HARVEST": {
                                "lanes": 1,
                                "live_ready_lanes": 0,
                                "promotion_candidate_lanes": 0,
                                "reward_jpy": 420.0,
                                "live_ready_reward_jpy": 0.0,
                                "potential_reward_jpy": 0.0,
                                "top_issue_codes": [{"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 1}],
                                "top_live_blocker_codes": [
                                    {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 1}
                                ],
                                "top_blockers": [{"label": "forecast confidence below live floor", "count": 1}],
                            },
                            "RUNNER": {
                                "lanes": 0,
                                "live_ready_lanes": 0,
                                "promotion_candidate_lanes": 0,
                                "reward_jpy": 0.0,
                            },
                        },
                        "runner_candidate_diagnostics": {
                            "status": "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST",
                            "trend_candidate_lanes": 4,
                            "runner_qualified_lanes": 0,
                            "attached_harvest_lanes": 4,
                            "top_demotion_reasons": [
                                {
                                    "reason": "RANGE regime is not a clean runner trend",
                                    "count": 3,
                                }
                            ],
                            "top_issue_codes": [
                                {
                                    "code": "FORECAST_WATCH_ONLY",
                                    "count": 1,
                                }
                            ],
                            "top_live_blocker_codes": [
                                {
                                    "code": "TREND_MARKET_NOT_OPERATING_TREND",
                                    "count": 2,
                                }
                            ],
                        },
                        "perspective_alignment_diagnostics": {
                            "status": "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED",
                            "pair_direction_groups": 4,
                            "range_forecast_method_mismatch_groups": 1,
                            "range_forecast_method_mismatch_lanes": 2,
                            "range_forecast_method_mismatch_top": [
                                {
                                    "pair": "EUR_USD",
                                    "direction": "SHORT",
                                    "method_mismatch_lanes": 2,
                                    "method_mismatch_reward_jpy": 2800.0,
                                    "range_rotation_lanes": 1,
                                    "range_rotation_live_ready_lanes": 0,
                                    "range_rotation_top_live_blocker_codes": [
                                        {"code": "RANGE_ROTATION_BROADER_LOCATION_CHASE", "count": 1}
                                    ],
                                    "top_live_blocker_codes": [
                                        {"code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION", "count": 2}
                                    ],
                                }
                            ],
                        },
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        evidence = codes["TARGET_OPEN_NO_LIVE_READY_LANES"]["evidence"]
        dry_run_blockers = {item["message"]: item for item in evidence["dry_run_passed_live_readiness_blockers"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(evidence["status_counts"]["DRY_RUN_PASSED"], 1)
        self.assertEqual(evidence["opportunity_modes"]["HARVEST"]["lanes"], 1)
        self.assertEqual(evidence["opportunity_modes"]["HARVEST"]["reward_jpy"], 420.0)
        self.assertEqual(evidence["opportunity_modes"]["RUNNER"]["top_issue_codes"][0]["code"], "FORECAST_WATCH_ONLY")
        self.assertEqual(
            evidence["opportunity_modes"]["RUNNER"]["top_live_blocker_codes"][0]["code"],
            "TREND_MARKET_NOT_OPERATING_TREND",
        )
        runner_diagnostics = evidence["runner_candidate_diagnostics"]
        self.assertEqual(runner_diagnostics["status"], "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST")
        self.assertEqual(runner_diagnostics["trend_candidate_lanes"], 4)
        self.assertEqual(runner_diagnostics["runner_qualified_lanes"], 0)
        self.assertEqual(runner_diagnostics["top_demotion_reasons"][0]["reason"], "RANGE regime is not a clean runner trend")
        self.assertEqual(runner_diagnostics["top_issue_codes"][0]["code"], "FORECAST_WATCH_ONLY")
        self.assertEqual(
            runner_diagnostics["top_live_blocker_codes"][0]["code"],
            "TREND_MARKET_NOT_OPERATING_TREND",
        )
        perspective_diagnostics = evidence["perspective_alignment_diagnostics"]
        self.assertEqual(perspective_diagnostics["status"], "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED")
        self.assertEqual(perspective_diagnostics["range_forecast_method_mismatch_lanes"], 2)
        self.assertEqual(
            perspective_diagnostics["range_forecast_method_mismatch_top"][0]["range_rotation_top_live_blocker_codes"][0]["code"],
            "RANGE_ROTATION_BROADER_LOCATION_CHASE",
        )
        self.assertIn("runner candidates", report_text)
        self.assertIn("perspective alignment", report_text)
        self.assertIn("opportunity modes", report_text)
        self.assertIn("reward=`420.0`", report_text)
        self.assertIn("live_codes=`TREND_MARKET_NOT_OPERATING_TREND`", report_text)
        self.assertIn("RANGE_METHOD_MISMATCH_REPAIR_REQUIRED", report_text)
        self.assertIn("EUR_USD SHORT mismatch=2", report_text)
        self.assertIn("dry-run blocker families", report_text)
        self.assertIn("nearest dry-run lanes", report_text)
        self.assertIn("forecast gate reasons", report_text)
        self.assertIn("RUNNER_CANDIDATES_DEMOTED_TO_HARVEST", report_text)
        self.assertIn("RANGE regime is not a clean runner trend=3", report_text)
        self.assertIn("failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT", report_text)
        self.assertIn("HARVEST", report_text)
        self.assertIn("forecast=1", report_text)
        self.assertIn("rr=`1.180`", report_text)
        self.assertIn("reward=`420.000`", report_text)
        self.assertIn("liquidity_sweep_high DOWN", report_text)
        self.assertEqual(dry_run_blockers["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]["count"], 1)
        self.assertEqual(dry_run_blockers["STRATEGY_NOT_ELIGIBLE"]["count"], 1)
        self.assertNotIn("STRATEGY_PROFILE_MISSING", dry_run_blockers)
        forecast_diagnostics = evidence["dry_run_passed_forecast_gate_diagnostics"]
        self.assertEqual(forecast_diagnostics["reason_counts"][0]["count"], 1)
        self.assertIn("liquidity_sweep_high DOWN", forecast_diagnostics["reason_counts"][0]["reason"])
        lane_diagnostic = forecast_diagnostics["lanes"][0]
        self.assertEqual(lane_diagnostic["lane_id"], "failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT")
        self.assertEqual(lane_diagnostic["opportunity_mode"], "HARVEST")
        self.assertEqual(lane_diagnostic["opportunity_mode_reward_risk"], 1.18)
        self.assertEqual(lane_diagnostic["tp_target_intent"], "HARVEST")
        self.assertEqual(lane_diagnostic["chart_direction_bias"], "LONG")
        self.assertEqual(lane_diagnostic["forecast_confidence"], 0.311)
        self.assertTrue(lane_diagnostic["forecast_market_support_ok"])
        self.assertEqual(lane_diagnostic["forecast_market_support_best_hit_rate"], 0.82)
        self.assertEqual(lane_diagnostic["forecast_market_support_top_signal"]["name"], "liquidity_sweep_high")

    def test_same_side_unselected_projection_arbitration_becomes_forecast_repair_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "range_trader:EUR_JPY:LONG:RANGE_ROTATION",
                                "status": "DRY_RUN_PASSED",
                                "intent": {
                                    "pair": "EUR_JPY",
                                    "side": "LONG",
                                    "order_type": "LIMIT",
                                    "metadata": {
                                        "forecast_direction": "RANGE",
                                        "forecast_confidence": 0.944,
                                        "forecast_raw_confidence": 0.821,
                                        "chart_direction_bias": "LONG",
                                        "forecast_market_support": {
                                            "ok": False,
                                            "direction": "RANGE",
                                            "reason": "forecast RANGE has no executable direction; audited projection unselected",
                                            "unselected_projection_count": 2,
                                            "unselected_reason": (
                                                "liquidity_sweep_low UP audited hit_rate=1.00 "
                                                "samples=40 was unselected because forecast=RANGE"
                                            ),
                                            "unselected_signals": [
                                                {
                                                    "confidence": 0.9918,
                                                    "direction": "UP",
                                                    "hit_rate": 1.0,
                                                    "name": "liquidity_sweep_low",
                                                    "samples": 40,
                                                    "timeframe": "M15",
                                                    "rationale": "sell-side sweep target, fade LONG",
                                                },
                                                {
                                                    "confidence": 0.8123,
                                                    "direction": "UP",
                                                    "hit_rate": 0.78,
                                                    "name": "liquidity_sweep_low",
                                                    "samples": 18,
                                                    "timeframe": "M30",
                                                    "rationale": "higher-timeframe sell-side sweep target",
                                                }
                                            ],
                                        },
                                    },
                                },
                                "risk_issues": [],
                                "strategy_issues": [],
                                "live_strategy_issues": [],
                                "live_blockers": [],
                                "risk_metrics": {
                                    "reward_jpy": 1200.0,
                                    "reward_risk": 1.2,
                                },
                            }
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("TARGET_OPEN_NO_LIVE_READY_LANES", codes)
        finding = codes["FORECAST_ARBITRATION_UNSELECTED_PROJECTION_REPAIR_REQUIRED"]
        self.assertEqual(finding["priority"], "P1")
        diagnostics = finding["evidence"]["forecast_arbitration_diagnostics"]
        self.assertEqual(diagnostics["lane_count"], 1)
        self.assertEqual(diagnostics["same_side_lane_count"], 1)
        self.assertEqual(diagnostics["same_side_actionable_repair_lane_count"], 1)
        self.assertEqual(diagnostics["same_side_context_blocked_lane_count"], 0)
        self.assertEqual(diagnostics["opposite_side_lane_count"], 0)
        self.assertEqual(diagnostics["relation_counts"][0]["relation"], "same_side")
        self.assertEqual(diagnostics["direction_counts"][0]["direction"], "UP")
        self.assertEqual(diagnostics["signal_counts"][0]["signal"], "liquidity_sweep_low:UP")
        self.assertEqual(diagnostics["lanes"][0]["pair"], "EUR_JPY")
        self.assertEqual(diagnostics["lanes"][0]["top_unselected_signal_side"], "LONG")
        self.assertEqual(diagnostics["lanes"][0]["top_unselected_signal_relation"], "same_side")
        self.assertEqual(diagnostics["lanes"][0]["top_unselected_signal"]["hit_rate"], 1.0)
        no_live_ready_evidence = codes["TARGET_OPEN_NO_LIVE_READY_LANES"]["evidence"]
        self.assertEqual(no_live_ready_evidence["forecast_arbitration_diagnostics"]["lane_count"], 1)
        self.assertIn("forecast arbitration", report_text)
        self.assertIn("relations=`same_side=1`", report_text)
        self.assertIn("EUR_JPY LONG->liquidity_sweep_low UP", report_text)

    def test_same_side_unselected_projection_with_context_blockers_is_not_actionable_repair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "range_trader:EUR_JPY:LONG:RANGE_ROTATION",
                                "status": "DRY_RUN_PASSED",
                                "intent": {
                                    "pair": "EUR_JPY",
                                    "side": "LONG",
                                    "order_type": "LIMIT",
                                    "metadata": {
                                        "forecast_direction": "RANGE",
                                        "forecast_confidence": 0.7,
                                        "forecast_raw_confidence": 0.82,
                                        "chart_direction_bias": "SHORT",
                                        "forecast_market_support": {
                                            "ok": False,
                                            "direction": "RANGE",
                                            "reason": "forecast RANGE has no executable direction; audited projection unselected",
                                            "unselected_projection_count": 1,
                                            "unselected_reason": (
                                                "liquidity_sweep_low UP audited hit_rate=0.78 "
                                                "samples=18 was unselected because forecast=RANGE"
                                            ),
                                            "unselected_signals": [
                                                {
                                                    "confidence": 0.8123,
                                                    "direction": "UP",
                                                    "hit_rate": 0.78,
                                                    "name": "liquidity_sweep_low",
                                                    "samples": 18,
                                                    "timeframe": "M30",
                                                    "rationale": "higher-timeframe sell-side sweep target",
                                                }
                                            ],
                                        },
                                    },
                                },
                                "risk_issues": [
                                    {
                                        "code": "CHART_DIRECTION_CONFLICT",
                                        "message": "chart direction conflicts with entry side",
                                        "severity": "WARN",
                                    }
                                ],
                                "strategy_issues": [],
                                "live_strategy_issues": [
                                    {
                                        "code": "STRATEGY_PROFILE_MISSING",
                                        "message": "EUR_JPY LONG is absent from mined strategy profile",
                                        "severity": "WARN",
                                    }
                                ],
                                "live_blockers": [],
                                "risk_metrics": {
                                    "reward_jpy": 1200.0,
                                    "reward_risk": 1.2,
                                },
                            }
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertNotIn("FORECAST_ARBITRATION_UNSELECTED_PROJECTION_REPAIR_REQUIRED", codes)
        finding = codes["FORECAST_ARBITRATION_SAME_SIDE_CONTEXT_BLOCKED"]
        self.assertEqual(finding["priority"], "P2")
        diagnostics = finding["evidence"]["forecast_arbitration_diagnostics"]
        self.assertEqual(diagnostics["same_side_lane_count"], 1)
        self.assertEqual(diagnostics["same_side_actionable_repair_lane_count"], 0)
        self.assertEqual(diagnostics["same_side_context_blocked_lane_count"], 1)
        self.assertEqual(
            diagnostics["same_side_context_blocked_lanes"][0]["context_blocker_families"],
            ["market_structure", "strategy_profile"],
        )
        self.assertEqual(
            diagnostics["same_side_context_blocker_counts"],
            [
                {"family": "market_structure", "count": 1},
                {"family": "strategy_profile", "count": 1},
            ],
        )
        self.assertIn("same_side_context_blocked=`1`", report_text)

    def test_opposite_unselected_projection_arbitration_is_enforced_not_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "range_trader:EUR_JPY:SHORT:RANGE_ROTATION",
                                "status": "DRY_RUN_PASSED",
                                "intent": {
                                    "pair": "EUR_JPY",
                                    "side": "SHORT",
                                    "order_type": "LIMIT",
                                    "metadata": {
                                        "forecast_direction": "RANGE",
                                        "forecast_confidence": 0.944,
                                        "forecast_raw_confidence": 0.821,
                                        "chart_direction_bias": "LONG",
                                        "forecast_market_support": {
                                            "ok": False,
                                            "direction": "RANGE",
                                            "reason": "forecast RANGE has no executable direction; audited projection unselected",
                                            "unselected_projection_count": 1,
                                            "unselected_reason": (
                                                "liquidity_sweep_low UP audited hit_rate=1.00 "
                                                "samples=40 was unselected because forecast=RANGE"
                                            ),
                                            "unselected_signals": [
                                                {
                                                    "confidence": 0.9918,
                                                    "direction": "UP",
                                                    "hit_rate": 1.0,
                                                    "name": "liquidity_sweep_low",
                                                    "samples": 40,
                                                    "timeframe": "M15",
                                                    "rationale": "sell-side sweep target, fade LONG",
                                                }
                                            ],
                                        },
                                    },
                                },
                                "risk_issues": [
                                    {
                                        "code": "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
                                        "message": "audited projection conflicts with RANGE lane",
                                        "severity": "WARN",
                                    }
                                ],
                                "strategy_issues": [],
                                "live_strategy_issues": [],
                                "live_blockers": [
                                    {
                                        "code": "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
                                        "message": "audited projection conflicts with RANGE lane",
                                        "severity": "BLOCK",
                                    }
                                ],
                                "risk_metrics": {
                                    "reward_jpy": 1200.0,
                                    "reward_risk": 1.2,
                                },
                            }
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertNotIn("FORECAST_ARBITRATION_UNSELECTED_PROJECTION_REPAIR_REQUIRED", codes)
        finding = codes["FORECAST_ARBITRATION_OPPOSITE_PROJECTION_CONFLICTS_ENFORCED"]
        self.assertEqual(finding["priority"], "P2")
        diagnostics = finding["evidence"]["forecast_arbitration_diagnostics"]
        self.assertEqual(diagnostics["lane_count"], 1)
        self.assertEqual(diagnostics["same_side_lane_count"], 0)
        self.assertEqual(diagnostics["opposite_side_lane_count"], 1)
        self.assertEqual(diagnostics["relation_counts"][0]["relation"], "opposite_side")
        self.assertEqual(diagnostics["lanes"][0]["top_unselected_signal_relation"], "opposite_side")
        self.assertIn("relations=`opposite_side=1`", report_text)

    def test_mixed_unselected_projection_arbitration_is_not_same_side_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "range_trader:GBP_JPY:SHORT:RANGE_ROTATION",
                                "status": "DRY_RUN_PASSED",
                                "intent": {
                                    "pair": "GBP_JPY",
                                    "side": "SHORT",
                                    "order_type": "LIMIT",
                                    "metadata": {
                                        "forecast_direction": "RANGE",
                                        "forecast_confidence": 0.985,
                                        "forecast_raw_confidence": 0.9,
                                        "chart_direction_bias": "LONG",
                                        "forecast_market_support": {
                                            "ok": False,
                                            "direction": "RANGE",
                                            "reason": "forecast RANGE has no executable direction; audited projection unselected",
                                            "unselected_projection_count": 2,
                                            "unselected_reason": "mixed sweep projections were unselected because forecast=RANGE",
                                            "unselected_signals": [
                                                {
                                                    "confidence": 0.68,
                                                    "direction": "DOWN",
                                                    "hit_rate": 0.825,
                                                    "name": "liquidity_sweep_high",
                                                    "samples": 40,
                                                    "timeframe": "M5",
                                                    "rationale": "buy-side sweep target, fade SHORT",
                                                },
                                                {
                                                    "confidence": 0.875,
                                                    "direction": "UP",
                                                    "hit_rate": 0.667,
                                                    "name": "liquidity_sweep_low",
                                                    "samples": 24,
                                                    "timeframe": "M15",
                                                    "rationale": "sell-side sweep target, fade LONG",
                                                },
                                            ],
                                        },
                                    },
                                },
                                "risk_issues": [
                                    {
                                        "code": "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
                                        "message": "audited projection conflicts with RANGE lane",
                                        "severity": "WARN",
                                    }
                                ],
                                "strategy_issues": [],
                                "live_strategy_issues": [],
                                "live_blockers": [
                                    {
                                        "code": "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
                                        "message": "audited projection conflicts with RANGE lane",
                                        "severity": "BLOCK",
                                    }
                                ],
                                "risk_metrics": {
                                    "reward_jpy": 1200.0,
                                    "reward_risk": 1.2,
                                },
                            }
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertNotIn("FORECAST_ARBITRATION_UNSELECTED_PROJECTION_REPAIR_REQUIRED", codes)
        finding = codes["FORECAST_ARBITRATION_OPPOSITE_PROJECTION_CONFLICTS_ENFORCED"]
        diagnostics = finding["evidence"]["forecast_arbitration_diagnostics"]
        self.assertEqual(diagnostics["lane_count"], 1)
        self.assertEqual(diagnostics["same_side_lane_count"], 0)
        self.assertEqual(diagnostics["opposite_side_lane_count"], 0)
        self.assertEqual(diagnostics["mixed_relation_lane_count"], 1)
        self.assertEqual(diagnostics["opposite_conflict_lane_count"], 1)
        self.assertEqual(diagnostics["relation_counts"][0]["relation"], "mixed_with_opposite")
        self.assertEqual(diagnostics["lanes"][0]["top_unselected_signal_relation"], "mixed_with_opposite")
        self.assertEqual(diagnostics["lanes"][0]["same_side_unselected_signal"]["direction"], "DOWN")
        self.assertEqual(diagnostics["lanes"][0]["opposite_side_unselected_signal"]["direction"], "UP")
        self.assertIn("relations=`mixed_with_opposite=1`", report_text)

    def test_coverage_perspective_mismatch_becomes_self_improvement_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["coverage"].write_text(
                json.dumps(
                    {
                        "status": "COVERAGE_GAP",
                        "perspective_alignment_diagnostics": {
                            "status": "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED",
                            "pair_direction_groups": 3,
                            "range_forecast_method_mismatch_groups": 1,
                            "range_forecast_method_mismatch_lanes": 5,
                            "range_forecast_method_mismatch_top": [
                                {
                                    "pair": "EUR_USD",
                                    "direction": "SHORT",
                                    "method_mismatch_lanes": 5,
                                    "method_mismatch_reward_jpy": 6500.0,
                                    "range_rotation_lanes": 2,
                                    "range_rotation_live_ready_lanes": 0,
                                    "range_rotation_top_live_blocker_codes": [
                                        {"code": "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT", "count": 2},
                                        {"code": "RANGE_ROTATION_BROADER_LOCATION_CHASE", "count": 2},
                                    ],
                                    "range_rotation_absence_reason": "OPPOSITE_RAIL_SIDE_SURFACED",
                                    "range_rotation_other_side_lanes": 1,
                                    "range_rotation_other_side_directions": [
                                        {"code": "LONG", "count": 1}
                                    ],
                                    "range_rotation_other_side_top_live_blocker_codes": [
                                        {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 1}
                                    ],
                                    "range_rotation_other_side_top_blockers": [
                                        {"label": "opposite rail confidence still below live floor", "count": 1}
                                    ],
                                    "top_live_blocker_codes": [
                                        {"code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION", "count": 5}
                                    ],
                                }
                            ],
                        },
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        finding = codes["RANGE_FORECAST_METHOD_MISMATCH_REPAIR_REQUIRED"]
        perspective = finding["evidence"]["perspective_alignment_diagnostics"]
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["layer"], "forecast")
        self.assertEqual(perspective["range_forecast_method_mismatch_lanes"], 5)
        self.assertEqual(perspective["range_forecast_method_mismatch_top"][0]["pair"], "EUR_USD")
        self.assertEqual(
            perspective["range_forecast_method_mismatch_top"][0]["range_rotation_top_live_blocker_codes"][0]["code"],
            "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
        )
        self.assertEqual(
            perspective["range_forecast_method_mismatch_top"][0]["range_rotation_other_side_directions"][0]["code"],
            "LONG",
        )
        self.assertEqual(
            perspective["range_forecast_method_mismatch_top"][0]["range_rotation_other_side_top_live_blocker_codes"][0]["code"],
            "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
        )
        self.assertIn("RANGE_FORECAST_METHOD_MISMATCH_REPAIR_REQUIRED", report_text)
        self.assertIn("perspective alignment", report_text)
        self.assertIn("EUR_USD SHORT mismatch=5", report_text)
        self.assertIn("other_rail=LONG", report_text)
        self.assertIn("other_blockers=FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", report_text)

    def test_partial_live_ready_coverage_still_names_target_shortfall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.5, pending_entry=True)
            files["target"].write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1000.0,
                        "remaining_minimum_jpy": 500.0,
                    }
                )
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "COVERAGE_GAP",
                        "remaining_target_jpy": 1000.0,
                        "live_ready_reward_jpy": 120.0,
                        "opportunity_modes": {
                            "HARVEST": {
                                "lanes": 4,
                                "live_ready_lanes": 1,
                                "live_ready_reward_jpy": 120.0,
                                "promotion_candidate_lanes": 0,
                                "top_issue_codes": [
                                    {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 3}
                                ],
                                "top_blockers": [
                                    {"label": "forecast confidence below live floor", "count": 3}
                                ],
                            },
                            "RUNNER": {
                                "lanes": 0,
                                "live_ready_lanes": 0,
                                "live_ready_reward_jpy": 0.0,
                                "promotion_candidate_lanes": 0,
                            },
                        },
                        "runner_candidate_diagnostics": {
                            "status": "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST",
                            "trend_candidate_lanes": 3,
                            "runner_qualified_lanes": 0,
                            "attached_harvest_lanes": 3,
                            "top_demotion_reasons": [
                                {"reason": "RANGE regime is not a clean runner trend", "count": 2}
                            ],
                        },
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        finding = codes["TARGET_OPEN_LIVE_READY_COVERAGE_SHORTFALL"]
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["live_ready_lanes"], 1)
        self.assertEqual(finding["evidence"]["live_ready_reward_jpy"], 120.0)
        self.assertEqual(finding["evidence"]["required_additional_reward_jpy"], 880.0)
        self.assertEqual(finding["evidence"]["minimum_floor_shortfall_jpy"], 380.0)
        self.assertEqual(finding["evidence"]["opportunity_modes"]["HARVEST"]["live_ready_lanes"], 1)
        self.assertEqual(
            finding["evidence"]["runner_candidate_diagnostics"]["status"],
            "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST",
        )
        self.assertIn("live coverage", report_text)
        self.assertIn("runner candidates", report_text)

    def test_lane_only_verification_blockers_are_not_p0_with_live_ready_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                verification_lane_blockers=True,
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("VERIFICATION_LEDGER_LANE_BLOCKERS_RECORDED", codes)
        self.assertNotIn("VERIFICATION_LEDGER_BLOCKED", codes)

    def test_persistent_profitability_discipline_escalates_to_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, -400.0, 50.0, -300.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "loss_side_market_close_count": 2,
                                "loss_side_market_close_net_jpy": -700.0,
                                "broker_trade_close_loss_side_market_close_count": 2,
                                "broker_trade_close_loss_side_market_close_source_counts": {
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 2
                                },
                                "broker_accepted_without_gateway_loss_side_market_close_count": 2,
                                "broker_accepted_without_gateway_loss_side_market_close_net_jpy": -700.0,
                                "broker_accepted_without_gateway_loss_side_market_close_source_counts": {
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 2
                                },
                                "broker_accepted_without_gateway_loss_side_market_close_evidence_counts": {
                                    "NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 2,
                                    "NO_CLIENT_EXTENSION": 2,
                                },
                            }
                        }
                    }
                )
            )

            first = _run(files, now=_NOW)
            retry = _run(files, now=_NOW + timedelta(seconds=30))
            second = _run(files, now=_NOW + timedelta(minutes=3))
            third = _run(files, now=_NOW + timedelta(minutes=6))
            payload = json.loads(files["output"].read_text())
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(first.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(retry.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(second.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(third.status, STATUS_BLOCKED)
        self.assertEqual(run_count, 3)
        self.assertIn("NEGATIVE_RECENT_EXPECTANCY", codes)
        self.assertIn("SMALL_WIN_LARGE_LOSS_ASYMMETRY", codes)
        self.assertIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        self.assertEqual(codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]["priority"], "P0")
        self.assertEqual(codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]["evidence"]["current_streak"], 3)
        close_evidence = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]["evidence"][
            "system_defect_evidence"
        ]["ai_backtest_close_gate_loss_evidence"]
        self.assertEqual(close_evidence["loss_side_market_close_count"], 2)
        self.assertEqual(close_evidence["broker_accepted_without_gateway_loss_side_market_close_count"], 2)
        self.assertEqual(
            close_evidence["broker_accepted_without_gateway_loss_side_market_close_evidence_counts"],
            {"NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 2, "NO_CLIENT_EXTENSION": 2},
        )

    def test_direct_manual_close_dominated_repaired_profitability_does_not_escalate_to_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_direct_manual_close_recovery_ledger(files["execution_db"])

            first = _run(files, now=_NOW)
            second = _run(files, now=_NOW + timedelta(minutes=3))
            third = _run(files, now=_NOW + timedelta(minutes=6))
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(first.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(second.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(third.status, STATUS_ACTION_REQUIRED)
        self.assertIn("NEGATIVE_RECENT_EXPECTANCY", codes)
        self.assertIn("SMALL_WIN_LARGE_LOSS_ASYMMETRY", codes)
        self.assertIn("DIRECT_OR_MANUAL_CLOSE_DOMINATED_PROFITABILITY_DRAG", codes)
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        repair = codes["DIRECT_OR_MANUAL_CLOSE_DOMINATED_PROFITABILITY_DRAG"]["evidence"]
        self.assertEqual(repair["non_gateway_close_drag_metric"]["net_jpy"], -700.0)
        self.assertEqual(repair["net_without_non_gateway_close_drag_jpy"], 150.0)
        self.assertEqual(repair["last_24h_non_gateway_market_close_loss_trades"], 0)

    @staticmethod
    def _failed_trailing_effect() -> dict:
        return {
            "closed_trades": 20,
            "net_jpy": -5000.0,
            "profit_factor": 0.4,
            "expectancy_jpy": -250.0,
            "avg_win_jpy": 300.0,
            "avg_loss_jpy_abs": 1200.0,
            "worst_segments": [],
            "close_provenance_metrics": {},
            "market_order_trade_close_loss_provenance_metrics": {},
        }

    @staticmethod
    def _positive_24h_gateway_effect() -> dict:
        # Mirrors the live 2026-06-11 deadlock evidence: trailing window still
        # negative, but the last-24h gateway-attributable closes are net
        # positive without loss asymmetry.
        return {
            "closed_trades": 4,
            "net_jpy": 366.1,
            "gross_profit_jpy": 788.5,
            "gross_loss_jpy": 422.4,
            "profit_factor": 1.87,
            "expectancy_jpy": 91.5,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 3,
                    "net_jpy": -282.7,
                    "gross_profit_jpy": 139.7,
                    "gross_loss_jpy": 422.4,
                    "win_trades": 1,
                    "loss_trades": 2,
                },
                "TAKE_PROFIT_ORDER": {
                    "trades": 1,
                    "net_jpy": 648.8,
                    "gross_profit_jpy": 648.8,
                    "gross_loss_jpy": 0.0,
                    "win_trades": 1,
                    "loss_trades": 0,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {},
        }

    @staticmethod
    def _bleeding_24h_gateway_effect() -> dict:
        return {
            "closed_trades": 3,
            "net_jpy": -7157.0,
            "gross_profit_jpy": 0.0,
            "gross_loss_jpy": 7157.0,
            "profit_factor": 0.0,
            "expectancy_jpy": -2385.7,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 3,
                    "net_jpy": -7157.0,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 7157.0,
                    "win_trades": 0,
                    "loss_trades": 3,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {},
        }

    def test_gateway_close_recovery_observation_conditions(self) -> None:
        observation = _gateway_close_recovery_observation(self._positive_24h_gateway_effect())
        self.assertIsNotNone(observation)
        self.assertEqual(observation["gateway_win_trades"], 2)
        self.assertEqual(observation["gateway_loss_trades"], 2)
        self.assertAlmostEqual(observation["gateway_net_jpy"], 366.1, places=1)

        self.assertIsNone(_gateway_close_recovery_observation(self._bleeding_24h_gateway_effect()))
        self.assertIsNone(_gateway_close_recovery_observation({"error": "missing"}))
        self.assertIsNone(_gateway_close_recovery_observation({"close_provenance_metrics": {}}))

        manual_only = {
            "close_provenance_metrics": {
                "NON_TRADER_CLIENT_EXTENSION": {
                    "trades": 1,
                    "net_jpy": 22000.0,
                    "gross_profit_jpy": 22000.0,
                    "gross_loss_jpy": 0.0,
                    "win_trades": 1,
                    "loss_trades": 0,
                },
            },
        }
        self.assertIsNone(_gateway_close_recovery_observation(manual_only))

        asymmetric_but_positive = {
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 4,
                    "net_jpy": 100.0,
                    "gross_profit_jpy": 1500.0,
                    "gross_loss_jpy": 1400.0,
                    "win_trades": 3,
                    "loss_trades": 1,
                },
            },
        }
        self.assertIsNone(_gateway_close_recovery_observation(asymmetric_but_positive))

    def test_persistent_profitability_downgrades_to_recovery_on_clean_24h_gateway_window(self) -> None:
        findings = _profitability_findings(
            run_id="run-recovery",
            effect=self._failed_trailing_effect(),
            effect_24h=self._positive_24h_gateway_effect(),
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=5,
        )
        codes = {item["code"]: item for item in findings}
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        self.assertIn("PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY", codes)
        recovery = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY"]
        self.assertEqual(recovery["priority"], "P1")
        self.assertEqual(recovery["evidence"]["current_streak"], 6)
        self.assertEqual(recovery["evidence"]["recovery_observation"]["gateway_win_trades"], 2)

    def test_persistent_profitability_stays_p0_when_24h_gateway_window_bleeds(self) -> None:
        findings = _profitability_findings(
            run_id="run-bleeding",
            effect=self._failed_trailing_effect(),
            effect_24h=self._bleeding_24h_gateway_effect(),
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=5,
        )
        codes = {item["code"]: item for item in findings}
        self.assertIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY", codes)
        self.assertEqual(codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]["priority"], "P0")

    def test_persistent_profitability_stays_p0_without_material_gateway_recovery_proof(self) -> None:
        effect_24h = {
            "closed_trades": 1,
            "net_jpy": -661.5,
            "gross_profit_jpy": 0.0,
            "gross_loss_jpy": 661.5,
            "profit_factor": 0.0,
            "expectancy_jpy": -661.5,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 1,
                    "net_jpy": -661.5,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 661.5,
                    "win_trades": 0,
                    "loss_trades": 1,
                    "loss_containment_trades": 1,
                    "loss_containment_net_jpy": -661.5,
                    "loss_containment_avoided_loss_jpy": 800.0,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 1,
                    "net_jpy": -661.5,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 661.5,
                    "win_trades": 0,
                    "loss_trades": 1,
                }
            },
        }

        findings = _profitability_findings(
            run_id="run-no-recovery-proof",
            effect={
                **self._failed_trailing_effect(),
                "profit_factor": 0.891,
                "expectancy_jpy": -29.04,
                "avg_win_jpy": 509.21,
                "avg_loss_jpy_abs": 500.0,
            },
            effect_24h=effect_24h,
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=5,
        )

        codes = {item["code"]: item for item in findings}
        blocked = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]
        self.assertEqual(blocked["priority"], "P0")
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY", codes)
        evidence = blocked["evidence"]["system_defect_evidence"]
        self.assertIn("persistent_negative_expectancy_without_recovery", evidence)
        self.assertFalse(
            evidence["persistent_negative_expectancy_without_recovery"][
                "last_24h_gateway_recovery_proven"
            ]
        )

    def test_persistent_profitability_recovers_when_gateway_close_only_materially_contained_loss(
        self,
    ) -> None:
        effect_24h = {
            "closed_trades": 1,
            "net_jpy": -661.5,
            "gross_profit_jpy": 0.0,
            "gross_loss_jpy": 661.5,
            "profit_factor": 0.0,
            "expectancy_jpy": -661.5,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 1,
                    "net_jpy": -661.5,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 661.5,
                    "win_trades": 0,
                    "loss_trades": 1,
                    "loss_containment_trades": 1,
                    "loss_containment_net_jpy": -661.5,
                    "loss_containment_avoided_loss_jpy": 5782.5,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 1,
                    "net_jpy": -661.5,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 661.5,
                    "win_trades": 0,
                    "loss_trades": 1,
                }
            },
        }

        findings = _profitability_findings(
            run_id="run-contained-only",
            effect={
                **self._failed_trailing_effect(),
                "profit_factor": 0.891,
                "expectancy_jpy": -29.04,
                "avg_win_jpy": 509.21,
                "avg_loss_jpy_abs": 500.0,
            },
            effect_24h=effect_24h,
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=5,
        )

        codes = {item["code"]: item for item in findings}
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        recovery = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY"]
        observation = recovery["evidence"]["recovery_observation"]
        self.assertEqual(recovery["priority"], "P1")
        self.assertEqual(observation["recovery_basis"], "material_loss_containment")
        self.assertEqual(observation["gateway_win_trades"], 0)
        self.assertEqual(observation["gateway_loss_trades"], 0)
        self.assertEqual(observation["gateway_net_jpy"], 0.0)
        self.assertEqual(observation["gateway_raw_net_jpy"], -661.5)
        self.assertEqual(observation["loss_containment_trades"], 1)
        self.assertGreater(observation["loss_containment_avoided_loss_jpy"], 1323.0)

    def test_persistent_profitability_recovers_when_gateway_loss_close_contained_sl_risk(self) -> None:
        effect_24h = {
            "closed_trades": 2,
            "net_jpy": -200.0,
            "gross_profit_jpy": 800.0,
            "gross_loss_jpy": 1000.0,
            "profit_factor": 0.8,
            "expectancy_jpy": -100.0,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 1,
                    "net_jpy": -1000.0,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 1000.0,
                    "win_trades": 0,
                    "loss_trades": 1,
                    "loss_containment_trades": 1,
                    "loss_containment_net_jpy": -1000.0,
                    "loss_containment_avoided_loss_jpy": 2400.0,
                },
                "TAKE_PROFIT_ORDER": {
                    "trades": 1,
                    "net_jpy": 800.0,
                    "gross_profit_jpy": 800.0,
                    "gross_loss_jpy": 0.0,
                    "win_trades": 1,
                    "loss_trades": 0,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {},
        }

        findings = _profitability_findings(
            run_id="run-contained-close",
            effect=self._failed_trailing_effect(),
            effect_24h=effect_24h,
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=5,
        )

        codes = {item["code"]: item for item in findings}
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        recovery = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY"]
        observation = recovery["evidence"]["recovery_observation"]
        self.assertEqual(recovery["priority"], "P1")
        self.assertEqual(observation["recovery_basis"], "winning_close_window")
        self.assertEqual(observation["gateway_raw_net_jpy"], -200.0)
        self.assertEqual(observation["gateway_net_jpy"], 800.0)
        self.assertEqual(observation["gateway_loss_trades"], 0)
        self.assertEqual(observation["loss_containment_trades"], 1)

    def test_persistent_profitability_escalates_when_gateway_close_bleeds_without_loss_asymmetry(
        self,
    ) -> None:
        effect = {
            "closed_trades": 29,
            "net_jpy": -209.58,
            "profit_factor": 0.971,
            "expectancy_jpy": -7.23,
            "avg_win_jpy": 509.21,
            "avg_loss_jpy_abs": 489.24,
            "worst_segments": [],
            "market_order_trade_close_loss_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 15,
                    "net_jpy": -7338.58,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 7338.58,
                    "win_trades": 0,
                    "loss_trades": 15,
                }
            },
        }
        effect_24h = {
            "closed_trades": 6,
            "net_jpy": -673.02,
            "gross_profit_jpy": 1515.79,
            "gross_loss_jpy": 2188.81,
            "profit_factor": 0.693,
            "expectancy_jpy": -112.17,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 5,
                    "net_jpy": -1331.96,
                    "gross_profit_jpy": 856.85,
                    "gross_loss_jpy": 2188.81,
                    "win_trades": 2,
                    "loss_trades": 3,
                },
                "TAKE_PROFIT_ORDER": {
                    "trades": 1,
                    "net_jpy": 658.94,
                    "gross_profit_jpy": 658.94,
                    "gross_loss_jpy": 0.0,
                    "win_trades": 1,
                    "loss_trades": 0,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 3,
                    "net_jpy": -2188.81,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 2188.81,
                    "win_trades": 0,
                    "loss_trades": 3,
                }
            },
        }

        findings = _profitability_findings(
            run_id="run-gateway-bleed",
            effect=effect,
            effect_24h=effect_24h,
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=2,
        )

        codes = {item["code"]: item for item in findings}
        self.assertIn("NEGATIVE_RECENT_EXPECTANCY", codes)
        self.assertNotIn("SMALL_WIN_LARGE_LOSS_ASYMMETRY", codes)
        blocked = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]
        self.assertEqual(blocked["priority"], "P0")
        bleed = blocked["evidence"]["system_defect_evidence"]["gateway_close_bleed_observation"]
        self.assertAlmostEqual(bleed["gateway_net_jpy"], -673.02, places=2)
        self.assertEqual(bleed["gateway_loss_trades"], 3)

    def test_profitability_streak_counts_negative_expectancy_without_loss_asymmetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "history.db"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE self_improvement_audit_runs (
                        run_uid TEXT PRIMARY KEY,
                        ts_utc TEXT NOT NULL,
                        status TEXT NOT NULL,
                        output_path TEXT NOT NULL,
                        report_path TEXT NOT NULL,
                        window_hours REAL NOT NULL,
                        findings INTEGER NOT NULL,
                        p0_findings INTEGER NOT NULL,
                        p1_findings INTEGER NOT NULL,
                        p2_findings INTEGER NOT NULL,
                        closed_trades INTEGER NOT NULL,
                        net_jpy REAL NOT NULL,
                        profit_factor REAL,
                        expectancy_jpy REAL,
                        live_ready_lanes INTEGER NOT NULL,
                        open_trader_positions INTEGER NOT NULL,
                        inserted_at_utc TEXT NOT NULL
                    );
                    CREATE TABLE self_improvement_findings (
                        finding_uid TEXT PRIMARY KEY,
                        run_uid TEXT NOT NULL,
                        ts_utc TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        layer TEXT NOT NULL,
                        code TEXT NOT NULL,
                        message TEXT NOT NULL,
                        next_action TEXT NOT NULL,
                        evidence_json TEXT NOT NULL,
                        inserted_at_utc TEXT NOT NULL
                    );
                    """
                )
                for index in range(3):
                    ts = (_NOW - timedelta(minutes=index * 3)).isoformat()
                    run_uid = f"run-{index}"
                    conn.execute(
                        """
                        INSERT INTO self_improvement_audit_runs(
                            run_uid, ts_utc, status, output_path, report_path, window_hours,
                            findings, p0_findings, p1_findings, p2_findings, closed_trades,
                            net_jpy, profit_factor, expectancy_jpy, live_ready_lanes,
                            open_trader_positions, inserted_at_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_uid,
                            ts,
                            STATUS_ACTION_REQUIRED,
                            "out.json",
                            "report.md",
                            168.0,
                            1,
                            0,
                            1,
                            0,
                            29,
                            -209.58,
                            0.971,
                            -7.23,
                            1,
                            0,
                            ts,
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO self_improvement_findings(
                            finding_uid, run_uid, ts_utc, priority, layer, code,
                            message, next_action, evidence_json, inserted_at_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"finding-{index}",
                            run_uid,
                            ts,
                            "P1",
                            "profitability",
                            "NEGATIVE_RECENT_EXPECTANCY",
                            "recent outcome window is not profitable",
                            "repair profitability discipline",
                            "{}",
                            ts,
                        ),
                    )

            auditor = SelfImprovementAuditor(history_db_path=db_path)

            self.assertEqual(auditor._history_code_streak(PROFITABILITY_DISCIPLINE_CODES), 3)
            self.assertEqual(
                auditor._history_code_streak(
                    ("NEGATIVE_RECENT_EXPECTANCY", "SMALL_WIN_LARGE_LOSS_ASYMMETRY")
                ),
                0,
            )

    def test_effect_metrics_attributes_closed_pl_to_opening_lane_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_method_attribution_ledger(db_path)

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        by_method = {
            (item["pair"], item["side"], item["method"]): item
            for item in effect["worst_segments"]
        }
        rotation = by_method[("EUR_USD", "SHORT", "RANGE_ROTATION")]
        self.assertEqual(rotation["trades"], 2)
        self.assertAlmostEqual(rotation["net_jpy"], -700.0)
        self.assertEqual(
            rotation["lane_ids"],
            ["range_trader:EUR_USD:SHORT:RANGE_ROTATION"],
        )
        self.assertEqual(rotation["trade_ids"], ["T1", "T2"])
        trend = by_method[("EUR_USD", "SHORT", "TREND_CONTINUATION")]
        self.assertEqual(trend["trades"], 1)
        self.assertAlmostEqual(trend["net_jpy"], 120.0)

    def test_effect_metrics_classifies_trader_entry_market_order_loss_close_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_market_close_attribution_ledger(db_path, include_gateway_close=False)
            with sqlite3.connect(db_path) as conn:
                conn.execute("ALTER TABLE execution_events ADD COLUMN raw_json TEXT")
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "broker-close-accept",
                        (_NOW - timedelta(hours=1, minutes=10)).isoformat(),
                        "ORDER_ACCEPTED",
                        "",
                        "C42",
                        "",
                        "EUR_USD",
                        "SHORT",
                        1000,
                        None,
                        "TRADE_CLOSE",
                        json.dumps(
                            {
                                "id": "C42",
                                "reason": "TRADE_CLOSE",
                                "tradeClose": {"tradeID": "T42", "units": "ALL"},
                            }
                        ),
                    ),
                )

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        market_loss = effect["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["TRADER_ENTRY_LANE_ID"]["trades"], 1)
        self.assertAlmostEqual(market_loss["TRADER_ENTRY_LANE_ID"]["net_jpy"], -500.0)
        segment = effect["worst_segments"][0]
        self.assertEqual(
            segment["close_provenance_counts"],
            {"TRADER_ENTRY_LANE_ID": 1},
        )
        self.assertEqual(
            segment["close_provenance_net_jpy"],
            {"TRADER_ENTRY_LANE_ID": -500.0},
        )

    def test_effect_metrics_classifies_stale_gpt_close_satisfied_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_market_close_attribution_ledger(db_path, include_gateway_close=False)
            with sqlite3.connect(db_path) as conn:
                _add_raw_json_column(conn)
                _insert_broker_trade_close_accept(conn)
                _insert_stale_gpt_close_satisfied(conn)

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        market_loss = effect["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["STALE_GPT_CLOSE_SATISFIED"]["trades"], 1)
        self.assertAlmostEqual(market_loss["STALE_GPT_CLOSE_SATISFIED"]["net_jpy"], -500.0)
        self.assertNotIn("DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE", market_loss)
        segment = effect["worst_segments"][0]
        self.assertEqual(segment["close_provenance_counts"], {"STALE_GPT_CLOSE_SATISFIED": 1})

    def test_effect_metrics_classifies_gateway_market_order_loss_close_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_market_close_attribution_ledger(db_path, include_gateway_close=True)

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        market_loss = effect["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["GATEWAY_TRADE_CLOSE_SENT"]["trades"], 1)
        self.assertAlmostEqual(market_loss["GATEWAY_TRADE_CLOSE_SENT"]["net_jpy"], -500.0)

    def test_effect_metrics_marks_gateway_loss_close_contained_before_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_market_close_attribution_ledger(db_path, include_gateway_close=True)
            with sqlite3.connect(db_path) as conn:
                conn.execute("UPDATE execution_events SET price = 1.1000 WHERE event_uid = 'fill'")
                conn.execute("UPDATE execution_events SET price = 1.0980 WHERE event_uid = 'close'")
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason, price, sl
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "protection-sl",
                        (_NOW - timedelta(hours=2, minutes=50)).isoformat(),
                        "PROTECTION_CREATED",
                        "",
                        "SL42",
                        "T42",
                        "EUR_USD",
                        "",
                        None,
                        None,
                        "ON_FILL",
                        1.0950,
                        1.0950,
                    ),
                )

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        market_loss = effect["market_order_trade_close_loss_provenance_metrics"]["GATEWAY_TRADE_CLOSE_SENT"]
        close_metric = effect["close_provenance_metrics"]["GATEWAY_TRADE_CLOSE_SENT"]
        self.assertEqual(market_loss["trades"], 1)
        self.assertAlmostEqual(market_loss["net_jpy"], -500.0)
        self.assertEqual(market_loss["loss_containment_trades"], 1)
        self.assertGreater(market_loss["loss_containment_avoided_loss_jpy"], 0.0)
        self.assertEqual(close_metric["loss_containment_trades"], 1)

    def test_effect_metrics_matches_gateway_market_order_loss_close_by_order_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_market_close_attribution_ledger(db_path, include_gateway_close=False)
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "gw-close-order-only",
                        (_NOW - timedelta(hours=1, minutes=5)).isoformat(),
                        "GATEWAY_TRADE_CLOSE_SENT",
                        "",
                        "C42",
                        "",
                        "EUR_USD",
                        "",
                        None,
                        None,
                        "REVIEW_EXIT",
                    ),
                )

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        market_loss = effect["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["GATEWAY_TRADE_CLOSE_SENT"]["trades"], 1)
        self.assertAlmostEqual(market_loss["GATEWAY_TRADE_CLOSE_SENT"]["net_jpy"], -500.0)
        self.assertNotIn("NO_LOCAL_CLOSE_PROVENANCE", market_loss)

    def test_unattributed_market_order_close_is_p1_execution_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("UNATTRIBUTED_MARKET_ORDER_CLOSES", codes)
        finding = codes["UNATTRIBUTED_MARKET_ORDER_CLOSES"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["unattributed_loss_count"], 1)
        self.assertEqual(finding["evidence"]["examples"][0]["trade_id"], "T42")

    def test_stale_gpt_close_satisfied_is_separate_from_unattributed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)
            with sqlite3.connect(files["execution_db"]) as conn:
                _add_raw_json_column(conn)
                _insert_broker_trade_close_accept(conn)
                _insert_stale_gpt_close_satisfied(conn)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("UNATTRIBUTED_MARKET_ORDER_CLOSES", codes)
        finding = codes["STALE_GPT_CLOSE_SATISFIED_AFTER_BROKER_CLOSE"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["stale_gpt_close_satisfied_count"], 1)
        self.assertEqual(finding["evidence"]["examples"][0]["trade_id"], "T42")

    def test_unattributed_market_order_close_reports_broker_accept_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)
            with sqlite3.connect(files["execution_db"]) as conn:
                conn.execute("ALTER TABLE execution_events ADD COLUMN raw_json TEXT")
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "broker-close-accept",
                        (_NOW - timedelta(hours=1, minutes=10)).isoformat(),
                        "ORDER_ACCEPTED",
                        "",
                        "C42",
                        "",
                        "EUR_USD",
                        "SHORT",
                        1000,
                        None,
                        "TRADE_CLOSE",
                        json.dumps(
                            {
                                "id": "C42",
                                "type": "MARKET_ORDER",
                                "reason": "TRADE_CLOSE",
                                "tradeClose": {"tradeID": "T42", "units": "ALL"},
                            }
                        ),
                    ),
                )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        finding = codes["UNATTRIBUTED_MARKET_ORDER_CLOSES"]
        self.assertEqual(finding["evidence"]["broker_trade_close_accept_count"], 1)
        self.assertEqual(
            finding["evidence"]["broker_trade_close_accept_source_counts"],
            {"TRADER_ENTRY_LANE_ID": 1},
        )
        self.assertEqual(
            finding["evidence"]["examples"][0]["broker_trade_close_accept_sources"],
            ["TRADER_ENTRY_LANE_ID"],
        )

    def test_broker_trade_close_accept_uses_entry_lane_source_when_close_has_no_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)
            with sqlite3.connect(files["execution_db"]) as conn:
                _add_raw_json_column(conn)
                conn.execute(
                    """
                    UPDATE execution_events
                    SET lane_id = ?
                    WHERE event_uid = 'fill'
                    """,
                    ("trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",),
                )
                _insert_broker_trade_close_accept(conn)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        finding = codes["UNATTRIBUTED_MARKET_ORDER_CLOSES"]
        self.assertEqual(
            finding["evidence"]["broker_trade_close_accept_source_counts"],
            {"TRADER_ENTRY_LANE_ID": 1},
        )
        self.assertEqual(
            finding["evidence"]["examples"][0]["broker_trade_close_accept_sources"],
            ["TRADER_ENTRY_LANE_ID"],
        )
        market_loss = payload["effect_metrics"]["window"]["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["TRADER_ENTRY_LANE_ID"]["trades"], 1)
        self.assertNotIn("DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE", market_loss)

    def test_broker_trade_close_accept_uses_gateway_entry_receipt_source_when_fill_lane_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)
            with sqlite3.connect(files["execution_db"]) as conn:
                _add_raw_json_column(conn)
                _insert_broker_trade_close_accept(conn)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        finding = codes["UNATTRIBUTED_MARKET_ORDER_CLOSES"]
        self.assertEqual(
            finding["evidence"]["broker_trade_close_accept_source_counts"],
            {"TRADER_ENTRY_LANE_ID": 1},
        )
        market_loss = payload["effect_metrics"]["window"]["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["TRADER_ENTRY_LANE_ID"]["trades"], 1)
        self.assertNotIn("DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE", market_loss)

    def test_gateway_close_receipt_satisfies_market_order_close_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=True)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("UNATTRIBUTED_MARKET_ORDER_CLOSES", codes)

    def test_reconciled_gpt_close_satisfies_market_order_close_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(
                files["execution_db"],
                include_gateway_close=False,
                include_reconciled_close=True,
            )
            with sqlite3.connect(files["execution_db"]) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "gpt-close-accepted",
                        (_NOW - timedelta(hours=1, minutes=10)).isoformat(),
                        "GATEWAY_GPT_CLOSE_ACCEPTED",
                        "",
                        "",
                        "T42",
                        "EUR_USD",
                        "",
                        None,
                        None,
                        "GPT_CLOSE_ACCEPTED",
                    ),
                )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        effect = payload["effect_metrics"]["window"]
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("ACCEPTED_GPT_CLOSE_WITHOUT_POSITION_GATEWAY_RECEIPT", codes)
        self.assertNotIn("UNATTRIBUTED_MARKET_ORDER_CLOSES", codes)
        self.assertEqual(
            effect["market_order_trade_close_loss_provenance_metrics"][
                "GATEWAY_TRADE_CLOSE_RECONCILED"
            ]["trades"],
            1,
        )

    def test_accepted_gpt_close_without_position_gateway_is_separate_from_unattributed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)
            with sqlite3.connect(files["execution_db"]) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "gpt-close-accepted",
                        (_NOW - timedelta(hours=1, minutes=10)).isoformat(),
                        "GATEWAY_GPT_CLOSE_ACCEPTED",
                        "",
                        "",
                        "T42",
                        "EUR_USD",
                        "",
                        None,
                        None,
                        "GPT_CLOSE_ACCEPTED",
                    ),
                )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("UNATTRIBUTED_MARKET_ORDER_CLOSES", codes)
        finding = codes["ACCEPTED_GPT_CLOSE_WITHOUT_POSITION_GATEWAY_RECEIPT"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["accepted_gpt_close_without_position_gateway_count"], 1)
        self.assertEqual(finding["evidence"]["examples"][0]["trade_id"], "T42")

    def test_rejected_close_for_closed_trades_is_not_current_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["T1"]},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "CLOSE_THESIS_STILL_VALID",
                                "message": "CLOSE rejected for stale fixture trade",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        self.assertIn("STALE_GPT_CLOSE_BLOCKERS_FOR_CLOSED_TRADES", codes)
        self.assertEqual(codes["STALE_GPT_CLOSE_BLOCKERS_FOR_CLOSED_TRADES"]["priority"], "P1")

    def test_missing_legacy_trader_decision_is_not_p1_when_gpt_decision_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["trader"].unlink()

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("TRADER_DECISION_UNREADABLE", codes)
        self.assertNotIn("GPT_DECISION_UNREADABLE", codes)

    def test_missing_legacy_trader_decision_is_reported_when_gpt_decision_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].unlink()
            files["trader"].unlink()

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(codes["GPT_DECISION_UNREADABLE"]["priority"], "P0")
        self.assertEqual(codes["TRADER_DECISION_UNREADABLE"]["priority"], "P1")

    def test_rejected_non_close_receipt_blockers_are_not_current_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "REQUEST_EVIDENCE", "close_trade_ids": []},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "UNKNOWN_EVIDENCE_REF",
                                "message": "unknown evidence refs: option:skew:unknown",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        self.assertIn("LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS"]["priority"], "P1")
        self.assertEqual(codes["LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS"]["evidence"]["action"], "REQUEST_EVIDENCE")

    def test_rejected_stale_trade_receipt_is_not_stale_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "TRADE", "close_trade_ids": []},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "SELF_IMPROVEMENT_P0_BLOCKS_TRADE",
                                "message": "trade rejected by prior audit P0",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("LATEST_GPT_DECISION_STALE", codes)
        self.assertIn("LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS"]["priority"], "P1")

    def test_rejected_close_for_active_trade_remains_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True)
            files["entry_thesis"].write_text(
                json.dumps(
                    {
                        "trade_id": "T1",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "filled_at_utc": _NOW.isoformat(),
                    }
                )
                + "\n"
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["T1"]},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "CLOSE_THESIS_STILL_VALID",
                                "message": "CLOSE rejected for active fixture trade",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertGreaterEqual(summary.p0_findings, 1)
        self.assertIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        self.assertEqual(
            codes["LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES"]["evidence"]["active_close_trade_ids"],
            ["T1"],
        )

    def test_operator_auth_required_close_is_not_reported_as_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True)
            files["position_thesis"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "assessments": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "REVIEW_CLOSE",
                                "context_notes": [
                                    "invalidation hit with technical invalidation confirmed against LONG"
                                ],
                            }
                        ],
                    }
                )
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["T1"]},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "CLOSE_OPERATOR_AUTH_REQUIRED",
                                "message": "explicit Gate B is still missing for T1",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertGreaterEqual(summary.p0_findings, 1)
        self.assertIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        self.assertNotIn("OPEN_POSITION_CLOSE_EVIDENCE_UNRESOLVED", codes)
        finding = codes["OPEN_POSITION_CLOSE_OPERATOR_AUTH_REQUIRED"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["active_close_trade_ids"], ["T1"])

    def test_guardian_close_review_is_reported_as_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True)
            files["gpt"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "ACCEPTED",
                        "decision": {"action": "TRADE"},
                        "verification_issues": [],
                    }
                )
            )
            files["entry_thesis"].write_text(
                json.dumps(
                    {
                        "trade_id": "T1",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "filled_at_utc": _NOW.isoformat(),
                    }
                )
                + "\n"
            )
            files["position_guardian_management"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "action": "HOLD_PROTECTED",
                        "positions": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "action": "HOLD_PROTECTED",
                                "close_review_action": "REVIEW_EXIT",
                                "reasons": [
                                    "loss-cut: structural OB broken across 2 TFs (M15@1.17000, H1@1.17100) (-120 JPY)",
                                    "QR_DISABLE_AUTO_CLOSE=1 -> REVIEW_EXIT demoted to HOLD_PROTECTED",
                                ],
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertGreaterEqual(summary.p1_findings, 1)
        self.assertIn("OPEN_POSITION_CLOSE_EVIDENCE_UNRESOLVED", codes)
        signals = codes["OPEN_POSITION_CLOSE_EVIDENCE_UNRESOLVED"]["evidence"]["signals"]
        self.assertEqual(signals[0]["source"], "position_guardian_management")
        self.assertEqual(signals[0]["trade_id"], "T1")

    def test_operator_auth_required_close_with_hold_sidecars_is_not_decision_history_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True)
            files["entry_thesis"].write_text(
                json.dumps(
                    {
                        "trade_id": "T1",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "filled_at_utc": _NOW.isoformat(),
                    }
                )
                + "\n"
            )
            files["position_thesis"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "assessments": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "REVIEW_CLOSE",
                                "context_notes": [
                                    "invalidation hit with technical invalidation confirmed against LONG"
                                ],
                            }
                        ],
                    }
                )
            )
            files["thesis_evolution"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "evolutions": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "status": "WEAKENED",
                                "verdict": "HOLD",
                                "rationale": "current forecast still supports the open position side",
                            }
                        ],
                    }
                )
            )
            files["forecast_persistence"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "verdicts": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "EXTEND",
                                "reason": "recent forecasts still support the open position side",
                            }
                        ],
                    }
                )
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["T1"]},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "CLOSE_OPERATOR_AUTH_REQUIRED",
                                "message": "explicit Gate B is still missing for T1",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        finding = codes["LATEST_GPT_DECISION_SOFT_CLOSE_ADVISORY_REJECTED"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["active_close_trade_ids"], ["T1"])
        self.assertIn("OPEN_POSITION_CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_operator_auth_required_close_with_opposite_side_sidecars_stays_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True)
            files["position_thesis"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "assessments": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "REVIEW_CLOSE",
                                "context_notes": [
                                    "invalidation hit with technical invalidation confirmed against LONG"
                                ],
                            }
                        ],
                    }
                )
            )
            files["thesis_evolution"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "evolutions": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "HOLD",
                                "rationale": "opposite side support must not protect the active LONG",
                            }
                        ],
                    }
                )
            )
            files["forecast_persistence"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "verdicts": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "EXTEND",
                                "reason": "opposite side support must not protect the active LONG",
                            }
                        ],
                    }
                )
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["T1"]},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "CLOSE_OPERATOR_AUTH_REQUIRED",
                                "message": "explicit Gate B is still missing for T1",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertGreaterEqual(summary.p0_findings, 1)
        self.assertIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        self.assertNotIn("LATEST_GPT_DECISION_SOFT_CLOSE_ADVISORY_REJECTED", codes)

    def test_accepted_wait_predating_snapshot_without_risk_is_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("LATEST_GPT_DECISION_STALE", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["priority"], "P1")
        self.assertEqual(
            codes["LATEST_GPT_DECISION_STALE"]["evidence"]["snapshot_fetched_at_utc"],
            _NOW.isoformat(),
        )
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["current_streak"], 1)

    def test_accepted_wait_predating_snapshot_with_open_position_stays_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=True,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["entry_thesis"].write_text(
                json.dumps(
                    {
                        "trade_id": "T1",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "filled_at_utc": _NOW.isoformat(),
                    }
                )
                + "\n"
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("LATEST_GPT_DECISION_STALE", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["priority"], "P0")

    def test_consumed_wait_decision_predating_snapshot_is_not_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )
            with sqlite3.connect(files["execution_db"]) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(event_uid, ts_utc, event_type, pair, side, realized_pl_jpy)
                    VALUES (?, ?, 'GATEWAY_ORDER_NO_ACTION', NULL, NULL, NULL)
                    """,
                    ("consumed-wait", (_NOW - timedelta(seconds=30)).isoformat()),
                )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("LATEST_GPT_DECISION_STALE", codes)

    def test_consumed_trade_decision_predating_snapshot_is_not_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {
                            "action": "TRADE",
                            "selected_lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                            "selected_lane_ids": ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"],
                        },
                        "verification_issues": [],
                    }
                )
            )
            with sqlite3.connect(files["execution_db"]) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(event_uid, ts_utc, event_type, pair, side, realized_pl_jpy)
                    VALUES (?, ?, 'GATEWAY_ORDER_SENT', 'EUR_USD', 'LONG', NULL)
                    """,
                    ("consumed-trade", (_NOW - timedelta(seconds=30)).isoformat()),
                )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("LATEST_GPT_DECISION_STALE", codes)

    def test_accepted_trade_consumed_by_current_pending_entry_is_not_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=True,
                pending_entry=True,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {
                            "action": "TRADE",
                            "selected_lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                            "selected_lane_ids": ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"],
                        },
                        "verification_issues": [],
                    }
                )
            )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("LATEST_GPT_DECISION_STALE", codes)

    def test_accepted_request_evidence_predating_snapshot_without_risk_is_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "REQUEST_EVIDENCE"},
                        "verification_issues": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("LATEST_GPT_DECISION_STALE", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["priority"], "P1")
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["live_ready_lanes"], 0)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["pending_entry_orders"], 0)

    def test_stale_gpt_decision_finding_records_history_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )

            _run(files, now=_NOW)
            _run(files, now=_NOW + timedelta(minutes=1))
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["current_streak"], 2)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["previous_streak"], 1)

    def test_stale_gpt_decision_is_not_p0_when_live_ready_entry_needs_fresh_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("LATEST_GPT_DECISION_STALE", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["priority"], "P1")
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["live_ready_lanes"], 1)

    def test_position_management_stale_uses_source_snapshot_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=True,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["position_management"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": (_NOW + timedelta(seconds=5)).isoformat(),
                        "snapshot_fetched_at_utc": (_NOW - timedelta(minutes=5)).isoformat(),
                        "positions": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("POSITION_MANAGEMENT_STALE", codes)
        self.assertEqual(codes["POSITION_MANAGEMENT_STALE"]["priority"], "P0")
        self.assertEqual(
            codes["POSITION_MANAGEMENT_STALE"]["evidence"]["sidecar_snapshot_fetched_at_utc"],
            (_NOW - timedelta(minutes=5)).isoformat(),
        )

    def test_cli_writes_audit_and_returns_blocked_code_for_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True, write_memory=False)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "self-improvement-audit",
                        "--db",
                        str(files["execution_db"]),
                        "--history-db",
                        str(files["history_db"]),
                        "--output",
                        str(files["output"]),
                        "--report",
                        str(files["report"]),
                        "--snapshot",
                        str(files["snapshot"]),
                        "--target-state",
                        str(files["target"]),
                        "--order-intents",
                        str(files["intents"]),
                        "--market-context-matrix",
                        str(files["market_context_matrix"]),
                        "--memory-health",
                        str(files["memory"]),
                        "--learning-audit",
                        str(files["learning"]),
                        "--ai-test-bot-backtest",
                        str(files["ai_backtest"]),
                        "--verification-ledger",
                        str(files["verification"]),
                        "--attack-advice",
                        str(files["attack_advice"]),
                        "--forecast-history",
                        str(files["forecast_history"]),
                        "--projection-ledger",
                        str(files["projection_ledger"]),
                        "--entry-thesis-ledger",
                        str(files["entry_thesis"]),
                        "--gpt-decision",
                        str(files["gpt"]),
                        "--trader-decision",
                        str(files["trader"]),
                        "--position-management",
                        str(files["position_management"]),
                        "--thesis-evolution",
                        str(files["thesis_evolution"]),
                        "--position-thesis",
                        str(files["position_thesis"]),
                        "--forecast-persistence",
                        str(files["forecast_persistence"]),
                        "--coverage-optimization",
                        str(files["coverage"]),
                    ]
                )

            result = json.loads(stdout.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], STATUS_BLOCKED)
            self.assertTrue(files["output"].exists())
            self.assertTrue(files["report"].exists())
            payload = json.loads(files["output"].read_text())
            self.assertEqual(payload["artifact_paths"]["ai_attack_advice"], str(files["attack_advice"]))


_NOW = datetime(2026, 6, 5, 0, 0, tzinfo=timezone.utc)


def _run(files: dict[str, Path], *, now: datetime = _NOW):
    return SelfImprovementAuditor(
        db_path=files["execution_db"],
        history_db_path=files["history_db"],
        output_path=files["output"],
        report_path=files["report"],
    ).run(
        snapshot_path=files["snapshot"],
        target_state_path=files["target"],
        order_intents_path=files["intents"],
        market_context_matrix_path=files["market_context_matrix"],
        memory_health_path=files["memory"],
        learning_audit_path=files["learning"],
        ai_test_bot_backtest_path=files["ai_backtest"],
        verification_ledger_path=files["verification"],
        forecast_history_path=files["forecast_history"],
        projection_ledger_path=files["projection_ledger"],
        entry_thesis_ledger_path=files["entry_thesis"],
        gpt_decision_path=files["gpt"],
        trader_decision_path=files["trader"],
        position_management_path=files["position_management"],
        position_guardian_management_path=files["position_guardian_management"],
        thesis_evolution_path=files["thesis_evolution"],
        position_thesis_path=files["position_thesis"],
        forecast_persistence_path=files["forecast_persistence"],
        coverage_optimization_path=files["coverage"],
        attack_advice_path=files["attack_advice"],
        now=now,
    )


def _fixtures(
    root: Path,
    *,
    active_position: bool,
    write_memory: bool = True,
    projection_expired: bool = False,
    live_ready_market_rr: float | None = None,
    unrealized_pl_jpy: float = 0.0,
    closed_pls: tuple[float, ...] = (100.0, -250.0, 50.0),
    verification_lane_blockers: bool = False,
    pending_entry: bool = False,
) -> dict[str, Path]:
    files = {
        "execution_db": root / "execution_ledger.db",
        "history_db": root / "self_improvement_history.db",
        "output": root / "self_improvement.json",
        "report": root / "self_improvement.md",
        "snapshot": root / "broker_snapshot.json",
        "target": root / "daily_target_state.json",
        "intents": root / "order_intents.json",
        "market_context_matrix": root / "market_context_matrix.json",
        "memory": root / "memory_health.json",
        "learning": root / "learning_audit.json",
        "ai_backtest": root / "ai_test_bot_backtest.json",
        "verification": root / "verification_ledger.json",
        "forecast_history": root / "forecast_history.jsonl",
        "projection_ledger": root / "projection_ledger.jsonl",
        "entry_thesis": root / "entry_thesis_ledger.jsonl",
        "gpt": root / "gpt_trader_decision.json",
        "trader": root / "trader_decision.json",
        "position_management": root / "position_management.json",
        "position_guardian_management": root / "position_guardian_management.json",
        "thesis_evolution": root / "thesis_evolution_report.json",
        "position_thesis": root / "position_thesis_report.json",
        "forecast_persistence": root / "forecast_persistence_report.json",
        "coverage": root / "coverage_optimization.json",
        "attack_advice": root / "ai_attack_advice.json",
    }
    positions = []
    if active_position:
        positions.append(
            {
                "trade_id": "T1",
                "pair": "EUR_USD",
                "side": "LONG",
                "owner": "trader",
                "units": 1000,
                "entry_price": 1.17,
                "take_profit": 1.18,
                "unrealized_pl_jpy": -120.0,
            }
        )
    orders = []
    if pending_entry:
        orders.append(
            {
                "order_id": "P1",
                "pair": "EUR_USD",
                "order_type": "STOP",
                "state": "PENDING",
                "units": 1000,
                "price": 1.171,
                "owner": "trader",
                "trade_id": None,
            }
        )
    files["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": _NOW.isoformat(),
                "account": {
                    "fetched_at_utc": _NOW.isoformat(),
                    "last_transaction_id": "100",
                    "unrealized_pl_jpy": unrealized_pl_jpy,
                },
                "positions": positions,
                "orders": orders,
                "quotes": {"EUR_USD": {"bid": 1.1701, "ask": 1.1702, "timestamp_utc": _NOW.isoformat()}},
            }
        )
    )
    files["target"].write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0}))
    results = []
    if live_ready_market_rr is not None:
        results.append(
            {
                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                "status": "LIVE_READY",
                "intent": {
                    "pair": "EUR_USD",
                    "order_type": "MARKET",
                    "metadata": {"order_timing": "NOW_MARKET"},
                },
                "risk_metrics": {"reward_risk": live_ready_market_rr},
                "risk_issues": [],
                "strategy_issues": [],
                "live_blockers": [],
            }
        )
    files["intents"].write_text(json.dumps({"results": results}))
    if write_memory:
        files["memory"].write_text(
            json.dumps(
                {
                    "generated_at_utc": _NOW.isoformat(),
                    "status": "MEMORY_HEALTH_PASS",
                    "issues": [],
                    "blockers": [],
                    "warnings": [],
                }
            )
        )
    files["learning"].write_text(
        json.dumps(
            {
                "status": "LEARNING_AUDIT_PASS",
                "blockers": [],
                "warnings": [],
                "learning_influence": {"influenced_lanes": 0},
                "effect_metrics": {"closed_trades": len(closed_pls)},
                "min_effect_sample": 3,
            }
        )
    )
    verification_payload: dict[str, object] = {"status": "OK", "blocking_observations": 0, "blocking_evidence": []}
    if verification_lane_blockers:
        verification_payload = {
            "status": "BLOCKED",
            "blocking_observations": 1,
            "blocking_evidence": [
                {
                    "source": "order_intents",
                    "source_path": str(files["intents"]),
                    "subject_type": "lane",
                    "subject_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                    "check_name": "lane_blockers",
                    "status": "BLOCK",
                    "severity": "BLOCK",
                    "evidence": {
                        "blockers": [
                            {
                                "code": "FRESH_ENTRY_REWARD_RISK_NOT_POSITIVE",
                                "message": "fresh entry reward/risk does not exceed 1.00x",
                                "severity": "BLOCK",
                            }
                        ]
                    },
                }
            ],
        }
    files["verification"].write_text(json.dumps(verification_payload))
    files["forecast_history"].write_text(
        json.dumps({"timestamp_utc": _NOW.isoformat(), "cycle_id": "cycle-1", "pair": "EUR_USD", "direction": "UP"})
        + "\n"
    )
    emitted = _NOW - timedelta(minutes=90 if projection_expired else 1)
    files["projection_ledger"].write_text(
        json.dumps(
            {
                "timestamp_emitted_utc": emitted.isoformat(),
                "pair": "EUR_USD",
                "signal_name": "directional_forecast",
                "resolution_window_min": 30.0,
                "resolution_status": "PENDING" if projection_expired else "HIT",
                "cycle_id": "cycle-1",
            }
        )
        + "\n"
    )
    if active_position:
        files["entry_thesis"].write_text("")
    else:
        files["entry_thesis"].write_text("")
    files["gpt"].write_text(json.dumps({"status": "ACCEPTED", "decision": {"action": "TRADE"}, "verification_issues": []}))
    files["trader"].write_text(json.dumps({"action": "SEND_ENTRY", "generated_at_utc": _NOW.isoformat()}))
    for key, list_key in (
        ("position_management", "positions"),
        ("thesis_evolution", "evolutions"),
        ("position_thesis", "assessments"),
        ("forecast_persistence", "verdicts"),
    ):
        files[key].write_text(json.dumps({"generated_at_utc": _NOW.isoformat(), list_key: []}))
    files["coverage"].write_text(json.dumps({"artifact_diagnostics": {}}))
    files["attack_advice"].write_text(
        json.dumps(
            {
                "generated_at_utc": _NOW.isoformat(),
                "status": "NO_ATTACK_ADVICE",
                "recommended_now_lane_ids": [],
            }
        )
    )
    _write_execution_ledger(files["execution_db"], closed_pls=closed_pls, last_transaction_id="100")
    return files


def _write_execution_ledger(path: Path, *, closed_pls: tuple[float, ...], last_transaction_id: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                pair TEXT,
                side TEXT,
                realized_pl_jpy REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", last_transaction_id, _NOW.isoformat()),
        )
        for idx, pl in enumerate(closed_pls):
            conn.execute(
                """
                INSERT INTO execution_events(event_uid, ts_utc, event_type, pair, side, realized_pl_jpy)
                VALUES (?, ?, 'TRADE_CLOSED', 'EUR_USD', 'LONG', ?)
                """,
                (f"closed-{idx}", (_NOW - timedelta(hours=idx + 1)).isoformat(), pl),
            )


def _write_pending_cancel_churn_ledger(path: Path) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                realized_pl_jpy REAL,
                exit_reason TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", "100", _NOW.isoformat()),
        )
        for idx, pair in enumerate(("AUD_CAD", "CAD_CHF", "NZD_CHF")):
            accepted_ts = _NOW - timedelta(minutes=30 - idx * 5)
            canceled_ts = accepted_ts + timedelta(minutes=8 + idx)
            order_id = f"O-pending-{idx}"
            lane_id = f"range_trader:{pair}:LONG:RANGE_ROTATION"
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, event_type, lane_id, order_id,
                    pair, side, units, price, exit_reason
                )
                VALUES (?, ?, 'ORDER_ACCEPTED', ?, ?, ?, 'LONG', 1000, ?, NULL)
                """,
                (f"accepted-{idx}", accepted_ts.isoformat(), lane_id, order_id, pair, 1.1 + idx / 1000),
            )
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, event_type, lane_id, order_id,
                    pair, side, units, price, exit_reason
                )
                VALUES (?, ?, 'ORDER_CANCELED', ?, ?, ?, 'LONG', 1000, ?, NULL)
                """,
                (f"canceled-{idx}", canceled_ts.isoformat(), lane_id, order_id, pair, 1.1 + idx / 1000),
            )


def _write_pending_mixed_cancel_churn_ledger(path: Path) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                realized_pl_jpy REAL,
                exit_reason TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", "100", _NOW.isoformat()),
        )
        for idx, pair in enumerate(("AUD_CAD", "CAD_CHF", "NZD_CHF", "EUR_CAD", "NZD_JPY")):
            accepted_ts = _NOW - timedelta(minutes=60 - idx * 5)
            resolved_ts = accepted_ts + timedelta(minutes=6 + idx)
            order_id = f"O-mixed-{idx}"
            lane_id = f"range_trader:{pair}:LONG:RANGE_ROTATION"
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, event_type, lane_id, order_id,
                    pair, side, units, price, exit_reason
                )
                VALUES (?, ?, 'ORDER_ACCEPTED', ?, ?, ?, 'LONG', 1000, ?, NULL)
                """,
                (f"accepted-mixed-{idx}", accepted_ts.isoformat(), lane_id, order_id, pair, 1.2 + idx / 1000),
            )
            event_type = "ORDER_FILLED" if idx < 2 else "ORDER_CANCELED"
            exit_reason = "LIMIT_ORDER" if event_type == "ORDER_FILLED" else None
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, event_type, lane_id, order_id,
                    pair, side, units, price, exit_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, 'LONG', 1000, ?, ?)
                """,
                (
                    f"resolved-mixed-{idx}",
                    resolved_ts.isoformat(),
                    event_type,
                    lane_id,
                    order_id,
                    pair,
                    1.2 + idx / 1000,
                    exit_reason,
                ),
            )


def _write_pending_replacement_churn_ledger(path: Path) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                realized_pl_jpy REAL,
                exit_reason TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", "100", _NOW.isoformat()),
        )
        rows = [
            (
                "accepted-source",
                _NOW - timedelta(minutes=55),
                "ORDER_ACCEPTED",
                "range_trader:AUD_CAD:SHORT:RANGE_ROTATION",
                "replace-source",
                "AUD_CAD",
                "SHORT",
                -1000,
                0.9898,
            ),
            (
                "canceled-source",
                _NOW - timedelta(minutes=50),
                "ORDER_CANCELED",
                "range_trader:AUD_CAD:SHORT:RANGE_ROTATION",
                "replace-source",
                "AUD_CAD",
                "SHORT",
                -1000,
                0.9898,
            ),
            (
                "accepted-next",
                _NOW - timedelta(minutes=45),
                "ORDER_ACCEPTED",
                "range_trader:AUD_CAD:SHORT:RANGE_ROTATION",
                "replace-next",
                "AUD_CAD",
                "SHORT",
                -1000,
                0.9897,
            ),
            (
                "accepted-orphan-1",
                _NOW - timedelta(minutes=40),
                "ORDER_ACCEPTED",
                "range_trader:CAD_CHF:LONG:RANGE_ROTATION",
                "orphan-1",
                "CAD_CHF",
                "LONG",
                1000,
                0.5665,
            ),
            (
                "canceled-orphan-1",
                _NOW - timedelta(minutes=32),
                "ORDER_CANCELED",
                "range_trader:CAD_CHF:LONG:RANGE_ROTATION",
                "orphan-1",
                "CAD_CHF",
                "LONG",
                1000,
                0.5665,
            ),
            (
                "accepted-orphan-2",
                _NOW - timedelta(minutes=25),
                "ORDER_ACCEPTED",
                "range_trader:NZD_JPY:SHORT:RANGE_ROTATION",
                "orphan-2",
                "NZD_JPY",
                "SHORT",
                -1000,
                93.56,
            ),
            (
                "canceled-orphan-2",
                _NOW - timedelta(minutes=18),
                "ORDER_CANCELED",
                "range_trader:NZD_JPY:SHORT:RANGE_ROTATION",
                "orphan-2",
                "NZD_JPY",
                "SHORT",
                -1000,
                93.56,
            ),
        ]
        for event_uid, ts, event_type, lane_id, order_id, pair, side, units, price in rows:
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, event_type, lane_id, order_id,
                    pair, side, units, price, exit_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (event_uid, ts.isoformat(), event_type, lane_id, order_id, pair, side, units, price),
            )


def _write_direct_manual_close_recovery_ledger(path: Path) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                tp REAL,
                sl REAL,
                realized_pl_jpy REAL,
                exit_reason TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", "100", _NOW.isoformat()),
        )
        rows = [
            (
                "fill-old-direct",
                (_NOW - timedelta(hours=49)).isoformat(),
                "ORDER_FILLED",
                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                "O-old",
                "T-old",
                "EUR_USD",
                "LONG",
                1000,
                None,
                None,
            ),
            (
                "close-old-direct",
                (_NOW - timedelta(hours=48)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C-old",
                "T-old",
                "EUR_USD",
                "LONG",
                1000,
                -700.0,
                "MARKET_ORDER_TRADE_CLOSE",
            ),
            (
                "fill-gateway-loss",
                (_NOW - timedelta(hours=3)).isoformat(),
                "ORDER_FILLED",
                "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE",
                "O-gw",
                "T-gw",
                "USD_CAD",
                "LONG",
                1000,
                None,
                None,
            ),
            (
                "gateway-close-sent",
                (_NOW - timedelta(hours=2, minutes=5)).isoformat(),
                "GATEWAY_TRADE_CLOSE_SENT",
                "",
                "",
                "T-gw",
                "USD_CAD",
                "",
                None,
                None,
                "GPT_CLOSE",
            ),
            (
                "close-gateway-loss",
                (_NOW - timedelta(hours=2)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C-gw",
                "T-gw",
                "USD_CAD",
                "LONG",
                1000,
                -50.0,
                "MARKET_ORDER_TRADE_CLOSE",
            ),
            (
                "close-tp-1",
                (_NOW - timedelta(hours=1)).isoformat(),
                "TRADE_CLOSED",
                "range_trader:EUR_CHF:LONG:RANGE_ROTATION",
                "TP-1",
                "T-tp-1",
                "EUR_CHF",
                "LONG",
                1000,
                100.0,
                "TAKE_PROFIT_ORDER",
            ),
            (
                "close-tp-2",
                (_NOW - timedelta(minutes=30)).isoformat(),
                "TRADE_CLOSED",
                "range_trader:EUR_GBP:SHORT:RANGE_ROTATION",
                "TP-2",
                "T-tp-2",
                "EUR_GBP",
                "SHORT",
                1000,
                100.0,
                "TAKE_PROFIT_ORDER",
            ),
        ]
        conn.executemany(
            """
            INSERT INTO execution_events(
                event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                pair, side, units, realized_pl_jpy, exit_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _write_method_attribution_ledger(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                realized_pl_jpy REAL
            );
            """
        )
        rows = [
            (
                "fill-range-1",
                (_NOW - timedelta(hours=6)).isoformat(),
                "ORDER_FILLED",
                "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                "O1",
                "T1",
                "EUR_USD",
                "SHORT",
                0.0,
            ),
            (
                "close-range-1",
                (_NOW - timedelta(hours=5)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C1",
                "T1",
                "EUR_USD",
                "SHORT",
                -500.0,
            ),
            (
                "fill-range-2",
                (_NOW - timedelta(hours=4)).isoformat(),
                "ORDER_FILLED",
                "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                "O2",
                "T2",
                "EUR_USD",
                "SHORT",
                0.0,
            ),
            (
                "close-range-2",
                (_NOW - timedelta(hours=3)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C2",
                "T2",
                "EUR_USD",
                "SHORT",
                -200.0,
            ),
            (
                "fill-trend",
                (_NOW - timedelta(hours=2)).isoformat(),
                "ORDER_FILLED",
                "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION",
                "O3",
                "T3",
                "EUR_USD",
                "SHORT",
                0.0,
            ),
            (
                "close-trend",
                (_NOW - timedelta(hours=1)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C3",
                "T3",
                "EUR_USD",
                "SHORT",
                120.0,
            ),
        ]
        conn.executemany(
            """
            INSERT INTO execution_events(
                event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                pair, side, realized_pl_jpy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _write_market_close_attribution_ledger(
    path: Path,
    *,
    include_gateway_close: bool,
    include_reconciled_close: bool = False,
) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                tp REAL,
                sl REAL,
                realized_pl_jpy REAL,
                exit_reason TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", "100", _NOW.isoformat()),
        )
        rows = [
            (
                "gw-entry",
                (_NOW - timedelta(hours=3)).isoformat(),
                "GATEWAY_ORDER_SENT",
                "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                "O42",
                "",
                "EUR_USD",
                "LONG",
                1000,
                None,
                None,
            ),
            (
                "fill",
                (_NOW - timedelta(hours=2, minutes=55)).isoformat(),
                "ORDER_FILLED",
                "",
                "O42",
                "T42",
                "EUR_USD",
                "LONG",
                1000,
                None,
                None,
            ),
        ]
        if include_gateway_close:
            rows.append(
                (
                    "gw-close",
                    (_NOW - timedelta(hours=1, minutes=5)).isoformat(),
                    "GATEWAY_TRADE_CLOSE_SENT",
                    "",
                    "",
                    "T42",
                    "EUR_USD",
                    "",
                    None,
                    None,
                    "GPT_CLOSE",
                )
            )
        if include_reconciled_close:
            rows.append(
                (
                    "gw-close-reconciled",
                    (_NOW - timedelta(hours=1, minutes=5)).isoformat(),
                    "GATEWAY_TRADE_CLOSE_RECONCILED",
                    "",
                    "C42",
                    "T42",
                    "EUR_USD",
                    "",
                    None,
                    None,
                    "GPT_CLOSE_RECONCILED",
                )
            )
        rows.append(
            (
                "close",
                (_NOW - timedelta(hours=1)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C42",
                "T42",
                "EUR_USD",
                "LONG",
                1000,
                -500.0,
                "MARKET_ORDER_TRADE_CLOSE",
            )
        )
        conn.executemany(
            """
            INSERT INTO execution_events(
                event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                pair, side, units, realized_pl_jpy, exit_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _add_raw_json_column(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE execution_events ADD COLUMN raw_json TEXT")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


def _insert_broker_trade_close_accept(
    conn: sqlite3.Connection,
    *,
    trade_id: str = "T42",
    order_id: str = "C42",
) -> None:
    conn.execute(
        """
        INSERT INTO execution_events(
            event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
            pair, side, units, realized_pl_jpy, exit_reason, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"broker-close-accept-{trade_id}",
            (_NOW - timedelta(hours=1, minutes=10)).isoformat(),
            "ORDER_ACCEPTED",
            "",
            order_id,
            "",
            "EUR_USD",
            "SHORT",
            1000,
            None,
            "TRADE_CLOSE",
            json.dumps(
                {
                    "id": order_id,
                    "reason": "TRADE_CLOSE",
                    "tradeClose": {"tradeID": trade_id, "units": "ALL"},
                }
            ),
        ),
    )


def _insert_stale_gpt_close_satisfied(
    conn: sqlite3.Connection,
    *,
    trade_id: str = "T42",
) -> None:
    conn.execute(
        """
        INSERT INTO execution_events(
            event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
            pair, side, units, realized_pl_jpy, exit_reason, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"stale-gpt-close-satisfied-{trade_id}",
            (_NOW - timedelta(minutes=55)).isoformat(),
            "GATEWAY_POSITION_NO_ACTION",
            "",
            "",
            trade_id,
            "EUR_USD",
            "",
            None,
            None,
            "GPT_CLOSE",
            json.dumps(
                {
                    "management_action": "GPT_CLOSE",
                    "sent": False,
                    "request": None,
                    "issues": [
                        {
                            "severity": "INFO",
                            "code": "STALE_CLOSE_ALREADY_ABSENT",
                            "message": "accepted CLOSE receipt named a trade id that is already absent",
                        }
                    ],
                }
            ),
        ),
    )
