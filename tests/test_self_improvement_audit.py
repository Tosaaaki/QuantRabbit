from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.self_improvement_audit import (
    PROJECTION_PENDING_EXPIRY_GRACE_SECONDS,
    STATUS_ACTION_REQUIRED,
    STATUS_BLOCKED,
    SelfImprovementAuditor,
    _effect_metrics,
    _gateway_close_recovery_observation,
    _intent_live_readiness_family_breakdown,
    _profitability_findings,
    _projection_expired,
    _top_intent_blockers,
    _top_intent_live_readiness_blockers,
)
from quant_rabbit.paths import DEFAULT_EXECUTION_LEDGER_DB, DEFAULT_SELF_IMPROVEMENT_HISTORY_DB


class SelfImprovementAuditorTest(unittest.TestCase):
    def test_default_history_db_is_separate_from_execution_ledger(self) -> None:
        auditor = SelfImprovementAuditor()

        self.assertEqual(auditor.db_path, DEFAULT_EXECUTION_LEDGER_DB)
        self.assertEqual(auditor.history_db_path, DEFAULT_SELF_IMPROVEMENT_HISTORY_DB)
        self.assertNotEqual(auditor.history_db_path, auditor.db_path)

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
                            }
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        evidence = codes["TARGET_OPEN_NO_LIVE_READY_LANES"]["evidence"]
        dry_run_blockers = {item["message"]: item for item in evidence["dry_run_passed_live_readiness_blockers"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(evidence["status_counts"]["DRY_RUN_PASSED"], 1)
        self.assertEqual(dry_run_blockers["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]["count"], 1)
        self.assertEqual(dry_run_blockers["STRATEGY_NOT_ELIGIBLE"]["count"], 1)
        self.assertNotIn("STRATEGY_PROFILE_MISSING", dry_run_blockers)
        forecast_diagnostics = evidence["dry_run_passed_forecast_gate_diagnostics"]
        self.assertEqual(forecast_diagnostics["reason_counts"][0]["count"], 1)
        self.assertIn("liquidity_sweep_high DOWN", forecast_diagnostics["reason_counts"][0]["reason"])
        lane_diagnostic = forecast_diagnostics["lanes"][0]
        self.assertEqual(lane_diagnostic["lane_id"], "failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT")
        self.assertEqual(lane_diagnostic["chart_direction_bias"], "LONG")
        self.assertEqual(lane_diagnostic["forecast_confidence"], 0.311)
        self.assertTrue(lane_diagnostic["forecast_market_support_ok"])
        self.assertEqual(lane_diagnostic["forecast_market_support_best_hit_rate"], 0.82)
        self.assertEqual(lane_diagnostic["forecast_market_support_top_signal"]["name"], "liquidity_sweep_high")

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

    def test_accepted_gpt_decision_predating_snapshot_is_p0(self) -> None:
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
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["priority"], "P0")
        self.assertEqual(
            codes["LATEST_GPT_DECISION_STALE"]["evidence"]["snapshot_fetched_at_utc"],
            _NOW.isoformat(),
        )
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["current_streak"], 1)

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
        thesis_evolution_path=files["thesis_evolution"],
        position_thesis_path=files["position_thesis"],
        forecast_persistence_path=files["forecast_persistence"],
        coverage_optimization_path=files["coverage"],
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
        "thesis_evolution": root / "thesis_evolution_report.json",
        "position_thesis": root / "position_thesis_report.json",
        "forecast_persistence": root / "forecast_persistence_report.json",
        "coverage": root / "coverage_optimization.json",
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
