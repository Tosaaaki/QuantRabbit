from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.learning_audit import LearningAuditor


class LearningAuditorTest(unittest.TestCase):
    def test_records_research_learning_influence_as_warn_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, ai_status="RESEARCH_PROFITABLE_NOT_CERTIFIED")
            db = root / "execution_ledger.db"
            _seed_execution_events(db)

            summary = LearningAuditor(
                db_path=db,
                output_path=root / "learning_audit.json",
                report_path=root / "learning_audit.md",
            ).run(
                ai_backtest_path=files["ai_backtest"],
                outcome_mart_path=files["outcome_mart"],
                post_trade_learning_path=files["post_trade_learning"],
                ai_attack_advice_path=files["ai_attack_advice"],
                now=datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(summary.status, "LEARNING_AUDIT_WARN")
            self.assertEqual(summary.blockers, 0)
            self.assertGreater(summary.warnings, 0)
            self.assertEqual(summary.influenced_lanes, 1)
            self.assertEqual(summary.total_learning_score_delta, 8.0)

            payload = json.loads((root / "learning_audit.json").read_text())
            self.assertEqual(payload["learning_influence"]["total_learning_score_delta"], 8.0)
            self.assertIn("Learning Audit Report", (root / "learning_audit.md").read_text())
            with sqlite3.connect(db) as conn:
                row = conn.execute(
                    "SELECT status, influenced_lanes, total_learning_score_delta FROM learning_audit_runs"
                ).fetchone()
            self.assertEqual(row, ("LEARNING_AUDIT_WARN", 1, 8.0))

    def test_allows_rank_only_projection_and_oanda_learning_influences_within_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, ai_status="RESEARCH_PROFITABLE_NOT_CERTIFIED")
            _write_rank_only_attack_advice(files["ai_attack_advice"])
            db = root / "execution_ledger.db"
            _seed_execution_events(db)

            summary = LearningAuditor(
                db_path=db,
                output_path=root / "learning_audit.json",
                report_path=root / "learning_audit.md",
            ).run(
                ai_backtest_path=files["ai_backtest"],
                outcome_mart_path=files["outcome_mart"],
                post_trade_learning_path=files["post_trade_learning"],
                ai_attack_advice_path=files["ai_attack_advice"],
                now=datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(summary.status, "LEARNING_AUDIT_WARN")
            self.assertEqual(summary.blockers, 0)
            self.assertEqual(summary.total_learning_score_delta, 34.0)
            payload = json.loads((root / "learning_audit.json").read_text())
            self.assertFalse(any("unknown or excessive score delta" in item for item in payload["blockers"]))

    def test_blocks_rank_only_projection_learning_influence_above_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, ai_status="RESEARCH_PROFITABLE_NOT_CERTIFIED")
            _write_rank_only_attack_advice(files["ai_attack_advice"], projection_delta=14.1)
            db = root / "execution_ledger.db"
            _seed_execution_events(db)

            summary = LearningAuditor(
                db_path=db,
                output_path=root / "learning_audit.json",
                report_path=root / "learning_audit.md",
            ).run(
                ai_backtest_path=files["ai_backtest"],
                outcome_mart_path=files["outcome_mart"],
                post_trade_learning_path=files["post_trade_learning"],
                ai_attack_advice_path=files["ai_attack_advice"],
                now=datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(summary.status, "LEARNING_AUDIT_BLOCKED")
            payload = json.loads((root / "learning_audit.json").read_text())
            self.assertTrue(any("unknown or excessive score delta" in item for item in payload["blockers"]))

    def test_blocks_research_influence_when_backtest_is_not_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, ai_status="BLOCKED")
            db = root / "execution_ledger.db"
            _seed_execution_events(db)

            summary = LearningAuditor(
                db_path=db,
                output_path=root / "learning_audit.json",
                report_path=root / "learning_audit.md",
            ).run(
                ai_backtest_path=files["ai_backtest"],
                outcome_mart_path=files["outcome_mart"],
                post_trade_learning_path=files["post_trade_learning"],
                ai_attack_advice_path=files["ai_attack_advice"],
                now=datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(summary.status, "LEARNING_AUDIT_BLOCKED")
            payload = json.loads((root / "learning_audit.json").read_text())
            self.assertTrue(any("research AI backtest influence" in item for item in payload["blockers"]))

    def test_blocks_positive_learning_influence_when_recent_effect_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, ai_status="RESEARCH_PROFITABLE_NOT_CERTIFIED")
            _write_learning_push_attack_advice(files["ai_attack_advice"])
            db = root / "execution_ledger.db"
            _seed_negative_execution_events(db)

            summary = LearningAuditor(
                db_path=db,
                output_path=root / "learning_audit.json",
                report_path=root / "learning_audit.md",
            ).run(
                ai_backtest_path=files["ai_backtest"],
                outcome_mart_path=files["outcome_mart"],
                post_trade_learning_path=files["post_trade_learning"],
                ai_attack_advice_path=files["ai_attack_advice"],
                min_effect_sample=2,
                now=datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(summary.status, "LEARNING_AUDIT_BLOCKED")
            payload = json.loads((root / "learning_audit.json").read_text())
            self.assertEqual(payload["learning_influence"]["risk_increasing_lanes"], 1)
            self.assertTrue(any("risk-increasing learning influence" in item for item in payload["blockers"]))

    def test_does_not_block_when_positive_learning_does_not_change_recommended_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, ai_status="RESEARCH_PROFITABLE_NOT_CERTIFIED")
            db = root / "execution_ledger.db"
            _seed_negative_execution_events(db)

            summary = LearningAuditor(
                db_path=db,
                output_path=root / "learning_audit.json",
                report_path=root / "learning_audit.md",
            ).run(
                ai_backtest_path=files["ai_backtest"],
                outcome_mart_path=files["outcome_mart"],
                post_trade_learning_path=files["post_trade_learning"],
                ai_attack_advice_path=files["ai_attack_advice"],
                min_effect_sample=2,
                now=datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(summary.status, "LEARNING_AUDIT_WARN")
            payload = json.loads((root / "learning_audit.json").read_text())
            self.assertEqual(payload["learning_influence"]["risk_increasing_lanes"], 0)
            self.assertFalse(any("risk-increasing learning influence" in item for item in payload["blockers"]))
            self.assertTrue(any("does not change the recommended risk set" in item for item in payload["warnings"]))

    def test_warns_for_protective_learning_penalty_when_recent_effect_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, ai_status="RESEARCH_PROFITABLE_NOT_CERTIFIED")
            _write_negative_attack_advice(files["ai_attack_advice"])
            db = root / "execution_ledger.db"
            _seed_negative_execution_events(db)

            summary = LearningAuditor(
                db_path=db,
                output_path=root / "learning_audit.json",
                report_path=root / "learning_audit.md",
            ).run(
                ai_backtest_path=files["ai_backtest"],
                outcome_mart_path=files["outcome_mart"],
                post_trade_learning_path=files["post_trade_learning"],
                ai_attack_advice_path=files["ai_attack_advice"],
                min_effect_sample=2,
                now=datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(summary.status, "LEARNING_AUDIT_WARN")
            self.assertEqual(summary.blockers, 0)
            self.assertEqual(summary.total_learning_score_delta, -15.0)
            payload = json.loads((root / "learning_audit.json").read_text())
            self.assertEqual(payload["learning_influence"]["risk_increasing_lanes"], 0)
            self.assertTrue(any("does not change the recommended risk set" in item for item in payload["warnings"]))

    def test_cli_runs_learning_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, ai_status="RESEARCH_PROFITABLE_NOT_CERTIFIED")
            db = root / "execution_ledger.db"
            _seed_execution_events(db)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "learning-audit",
                        "--db",
                        str(db),
                        "--output",
                        str(root / "learning_audit.json"),
                        "--report",
                        str(root / "learning_audit.md"),
                        "--ai-backtest",
                        str(files["ai_backtest"]),
                        "--outcome-mart",
                        str(files["outcome_mart"]),
                        "--post-trade-learning",
                        str(files["post_trade_learning"]),
                        "--ai-attack-advice",
                        str(files["ai_attack_advice"]),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "LEARNING_AUDIT_WARN")
            self.assertEqual(payload["influenced_lanes"], 1)

    def test_warns_when_market_order_trade_closes_drive_recent_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, ai_status="RESEARCH_PROFITABLE_NOT_CERTIFIED")
            db = root / "execution_ledger.db"
            _seed_exit_reason_drag_events(db)

            summary = LearningAuditor(
                db_path=db,
                output_path=root / "learning_audit.json",
                report_path=root / "learning_audit.md",
            ).run(
                ai_backtest_path=files["ai_backtest"],
                outcome_mart_path=files["outcome_mart"],
                post_trade_learning_path=files["post_trade_learning"],
                ai_attack_advice_path=files["ai_attack_advice"],
                now=datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(summary.status, "LEARNING_AUDIT_WARN")
            payload = json.loads((root / "learning_audit.json").read_text())
            market_close = payload["effect_metrics"]["exit_reason_metrics"]["MARKET_ORDER_TRADE_CLOSE"]
            self.assertEqual(market_close["closed_trades"], 3)
            self.assertEqual(market_close["net_jpy"], -600.0)
            self.assertTrue(any("market-order trade closes are negative" in item for item in payload["warnings"]))
            self.assertIn("MARKET_ORDER_TRADE_CLOSE", (root / "learning_audit.md").read_text())


def _fixtures(root: Path, *, ai_status: str) -> dict[str, Path]:
    files = {
        "ai_backtest": root / "ai_backtest.json",
        "outcome_mart": root / "outcome_mart.json",
        "post_trade_learning": root / "post_trade_learning.json",
        "ai_attack_advice": root / "ai_attack_advice.json",
    }
    blockers = [] if ai_status == "TARGET_COVERAGE_CERTIFIED" else ["not certified"]
    files["ai_backtest"].write_text(
        json.dumps(
            {
                "status": ai_status,
                "live_permission": False,
                "blockers": blockers,
                "training_days": 12,
                "min_train_trades": 5,
                "summary": {
                    "selected_trades": 40,
                    "total_managed_net_jpy": 2500.0 if ai_status != "BLOCKED" else -100.0,
                    "profit_factor": 1.4 if ai_status != "BLOCKED" else 0.8,
                },
            }
        )
    )
    files["outcome_mart"].write_text(
        json.dumps(
            {
                "read_only": True,
                "live_permission": False,
                "condition_validation": {
                    "status": "CONDITION_WALK_FORWARD_READY",
                    "validated_outcomes": 50,
                    "directional_hit_rate_pct": 60.0,
                },
            }
        )
    )
    files["post_trade_learning"].write_text(
        json.dumps({"status": "READY_FOR_REVIEW", "blockers": [], "profile_update_candidates": []})
    )
    files["ai_attack_advice"].write_text(
        json.dumps(
            {
                "status": "ATTACK_PARTIAL",
                "read_only": True,
                "live_permission": False,
                "recommended_now_lane_ids": ["lane:EUR_USD:LONG"],
                "lanes": [
                    {
                        "lane_id": "lane:EUR_USD:LONG",
                        "score": 42.0,
                        "learning_score_delta": 8.0,
                        "learning_influences": ["ai_backtest_research_positive_edge"],
                        "learning_influence_details": [
                            {
                                "source": "ai_backtest",
                                "influence": "ai_backtest_research_positive_edge",
                                "score_delta": 8.0,
                                "source_status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                                "edge_jpy": 5000.0,
                                "trades": 40,
                            }
                        ],
                    }
                ],
            }
        )
    )
    return files


def _write_negative_attack_advice(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "ATTACK_PARTIAL",
                "read_only": True,
                "live_permission": False,
                "recommended_now_lane_ids": ["lane:EUR_USD:LONG"],
                "lanes": [
                    {
                        "lane_id": "lane:EUR_USD:LONG",
                        "score": 42.0,
                        "learning_score_delta": -15.0,
                        "learning_influences": ["outcome_mart_negative_edge"],
                        "learning_influence_details": [
                            {
                                "source": "outcome_mart",
                                "influence": "outcome_mart_negative_edge",
                                "score_delta": -15.0,
                                "net_jpy": -1200.0,
                                "outcomes": 8,
                                "validation_outcomes": 2,
                            }
                        ],
                    }
                ],
            }
        )
    )


def _write_rank_only_attack_advice(path: Path, *, projection_delta: float = 14.0) -> None:
    oanda_delta = 20.0
    path.write_text(
        json.dumps(
            {
                "status": "ATTACK_PARTIAL",
                "read_only": True,
                "live_permission": False,
                "recommended_now_lane_ids": ["lane:EUR_USD:SHORT"],
                "lanes": [
                    {
                        "lane_id": "lane:EUR_USD:SHORT",
                        "score": 90.0,
                        "learning_score_delta": projection_delta + oanda_delta,
                        "learning_influences": [
                            "projection_economic_precision_rank_edge",
                            "oanda_universal_rotation_rank_edge",
                        ],
                        "learning_influence_details": [
                            {
                                "source": "projection_ledger",
                                "influence": "projection_economic_precision_rank_edge",
                                "score_delta": projection_delta,
                                "rank_only": True,
                                "signal_name": "bb_squeeze_expansion_imminent",
                                "economic_samples": 100,
                                "economic_hit_rate_wilson_lower": 0.9155,
                            },
                            {
                                "source": "oanda_universal_rotation",
                                "influence": "oanda_universal_rotation_rank_edge",
                                "score_delta": oanda_delta,
                                "raw_score_delta": oanda_delta,
                                "rank_only": True,
                                "rule_name": "EUR_USD_SHORT_M5_ROTATION",
                                "validation_samples": 24,
                                "validation_wilson95_lower": 0.72,
                            },
                        ],
                    }
                ],
            }
        )
    )


def _write_learning_push_attack_advice(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "ATTACK_PARTIAL",
                "read_only": True,
                "live_permission": False,
                "recommended_now_lane_ids": ["lane:EUR_USD:LONG"],
                "lanes": [
                    {
                        "lane_id": "lane:EUR_USD:LONG",
                        "score": 42.0,
                        "learning_score_delta": 8.0,
                        "learning_influences": ["ai_backtest_research_positive_edge"],
                        "learning_influence_details": [
                            {
                                "source": "ai_backtest",
                                "influence": "ai_backtest_research_positive_edge",
                                "score_delta": 8.0,
                                "source_status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                                "edge_jpy": 5000.0,
                                "trades": 40,
                            }
                        ],
                    },
                    {
                        "lane_id": "lane:GBP_USD:LONG",
                        "score": 40.0,
                        "learning_score_delta": 0.0,
                        "learning_influences": [],
                        "learning_influence_details": [],
                    },
                ],
            }
        )
    )


def _seed_execution_events(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                realized_pl_jpy REAL
            )
            """
        )
        conn.execute(
            "INSERT INTO execution_events VALUES ('2026-06-02T23:00:00+00:00', 'TRADE_CLOSED', 120.0)"
        )
        conn.execute(
            "INSERT INTO execution_events VALUES ('2026-06-02T22:00:00+00:00', 'TRADE_CLOSED', -80.0)"
        )


def _seed_negative_execution_events(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                realized_pl_jpy REAL
            )
            """
        )
        conn.executemany(
            "INSERT INTO execution_events VALUES (?, ?, ?)",
            [
                ("2026-06-02T23:00:00+00:00", "TRADE_CLOSED", -120.0),
                ("2026-06-02T22:00:00+00:00", "TRADE_CLOSED", 40.0),
            ],
        )


def _seed_exit_reason_drag_events(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                exit_reason TEXT,
                realized_pl_jpy REAL
            )
            """
        )
        rows = [
            ("2026-06-02T23:00:00.123456789Z", "TRADE_CLOSED", "TAKE_PROFIT_ORDER", 150.0),
            ("2026-06-02T22:30:00.123456789Z", "TRADE_CLOSED", "MARKET_ORDER_TRADE_CLOSE", -100.0),
            ("2026-06-02T22:00:00.123456789Z", "TRADE_CLOSED", "MARKET_ORDER_TRADE_CLOSE", -200.0),
            ("2026-06-02T21:30:00.123456789Z", "TRADE_CLOSED", "MARKET_ORDER_TRADE_CLOSE", -300.0),
        ]
        conn.executemany("INSERT INTO execution_events VALUES (?, ?, ?, ?)", rows)


if __name__ == "__main__":
    unittest.main()
