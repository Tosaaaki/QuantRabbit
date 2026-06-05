from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.memory_health import FORECAST_SNAPSHOT_GRACE, MemoryHealthAuditor, STATUS_BLOCKED, STATUS_PASS


class MemoryHealthAuditorTest(unittest.TestCase):
    def test_passes_when_all_memory_layers_are_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)

            summary = _run(files)

        self.assertEqual(summary.status, STATUS_PASS)
        self.assertEqual(summary.blockers, 0)
        self.assertEqual(summary.layers["short_term"], "PASS")
        self.assertEqual(summary.layers["medium_term"], "PASS")
        self.assertEqual(summary.layers["long_term"], "PASS")
        self.assertEqual(summary.layers["position_memory"], "PASS")

    def test_blocks_empty_strategy_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["strategy_profile"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "history_db": str(files["history_db"]),
                        "profiles": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(summary.layers["long_term"], "BLOCK")
        self.assertTrue(any(issue["code"] == "LONG_STRATEGY_PROFILE_EMPTY" for issue in payload["issues"]))

    def test_blocks_open_position_missing_entry_thesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, write_entry_thesis=False)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(summary.layers["position_memory"], "BLOCK")
        self.assertTrue(
            any(issue["code"] == "POSITION_ENTRY_THESIS_MISSING_FOR_OPEN_TRADE" for issue in payload["issues"])
        )

    def test_blocks_expired_pending_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, projection_expired=True)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(summary.layers["medium_term"], "BLOCK")
        self.assertTrue(any(issue["code"] == "MEDIUM_PROJECTION_LEDGER_EXPIRED_PENDING" for issue in payload["issues"]))

    def test_same_cycle_forecast_snapshot_drift_does_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            forecast_ts = _NOW - FORECAST_SNAPSHOT_GRACE + timedelta(seconds=1)
            _write_forecast_history(files["forecast_history"], forecast_ts)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        self.assertEqual(summary.status, STATUS_PASS)
        self.assertFalse(any(issue["code"] == "SHORT_FORECAST_HISTORY_STALE" for issue in payload["issues"]))
        self.assertFalse(any(issue["code"] == "SHORT_FORECAST_PAIR_STALE" for issue in payload["issues"]))

    def test_forecast_snapshot_drift_beyond_grace_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            forecast_ts = _NOW - FORECAST_SNAPSHOT_GRACE - timedelta(seconds=1)
            _write_forecast_history(files["forecast_history"], forecast_ts)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        self.assertEqual(summary.status, STATUS_BLOCKED)
        codes = {issue["code"] for issue in payload["issues"] if issue["severity"] == "BLOCK"}
        self.assertIn("SHORT_FORECAST_HISTORY_STALE", codes)
        self.assertIn("SHORT_FORECAST_PAIR_STALE", codes)

    def test_legacy_duplicate_forecast_and_projection_keys_remain_metrics_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            old_ts = (_NOW - timedelta(days=1)).isoformat()
            files["forecast_history"].write_text(
                files["forecast_history"].read_text()
                + json.dumps(
                    {
                        "timestamp_utc": old_ts,
                        "cycle_id": "old-cycle",
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.7,
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "timestamp_utc": old_ts,
                        "cycle_id": "old-cycle",
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.7,
                    }
                )
                + "\n"
            )
            old_projection = {
                "timestamp_emitted_utc": old_ts,
                "pair": "EUR_USD",
                "signal_name": "directional_forecast",
                "direction": "UP",
                "entry_price": 1.16,
                "predicted_target_price": 1.17,
                "resolution_window_min": 30.0,
                "resolution_status": "HIT",
                "cycle_id": "old-cycle",
            }
            files["projection_ledger"].write_text(
                files["projection_ledger"].read_text()
                + json.dumps(old_projection)
                + "\n"
                + json.dumps(old_projection)
                + "\n"
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        self.assertEqual(summary.status, STATUS_PASS)
        self.assertEqual(payload["metrics"]["forecast_history"]["duplicate_cycle_pairs"], 1)
        self.assertEqual(payload["metrics"]["forecast_history"]["current_duplicate_cycle_pairs"], 0)
        self.assertEqual(payload["metrics"]["projection_ledger"]["duplicate_projection_keys"], 1)
        self.assertEqual(payload["metrics"]["projection_ledger"]["current_duplicate_projection_keys"], 0)
        self.assertFalse(
            any(issue["code"] == "SHORT_FORECAST_HISTORY_DUPLICATE_CYCLE_PAIR" for issue in payload["issues"])
        )
        self.assertFalse(any(issue["code"] == "MEDIUM_PROJECTION_LEDGER_DUPLICATE_KEY" for issue in payload["issues"]))

    def test_current_duplicate_forecast_and_projection_keys_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            current_forecast = {
                "timestamp_utc": _NOW.isoformat(),
                "cycle_id": "cycle-1",
                "pair": "EUR_USD",
                "direction": "UP",
                "confidence": 0.8,
            }
            current_projection = {
                "timestamp_emitted_utc": _NOW.isoformat(),
                "pair": "EUR_USD",
                "signal_name": "directional_forecast",
                "direction": "UP",
                "entry_price": 1.17,
                "predicted_target_price": 1.18,
                "resolution_window_min": 30.0,
                "resolution_status": "HIT",
                "cycle_id": "cycle-1",
            }
            files["forecast_history"].write_text(files["forecast_history"].read_text() + json.dumps(current_forecast) + "\n")
            files["projection_ledger"].write_text(files["projection_ledger"].read_text() + json.dumps(current_projection) + "\n")

            _run(files)
            payload = json.loads(files["output"].read_text())

        self.assertEqual(payload["metrics"]["forecast_history"]["current_duplicate_cycle_pairs"], 1)
        self.assertEqual(payload["metrics"]["projection_ledger"]["current_duplicate_projection_keys"], 1)
        self.assertTrue(
            any(issue["code"] == "SHORT_FORECAST_HISTORY_DUPLICATE_CYCLE_PAIR" for issue in payload["issues"])
        )
        self.assertTrue(any(issue["code"] == "MEDIUM_PROJECTION_LEDGER_DUPLICATE_KEY" for issue in payload["issues"]))

    def test_live_ready_lane_keeps_advisory_profile_warnings_from_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                "status": "LIVE_READY",
                                "intent": {"pair": "EUR_USD"},
                                "risk_issues": [],
                                "strategy_issues": [
                                    {
                                        "code": "STRATEGY_PROFILE_MISSING",
                                        "message": "synthetic forecast-seed profile warning",
                                        "severity": "WARN",
                                    }
                                ],
                                "live_blockers": [],
                            },
                            {
                                "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                                "status": "DRY_RUN_BLOCKED",
                                "intent": {"pair": "EUR_USD"},
                                "risk_issues": [],
                                "strategy_issues": [],
                                "live_blockers": ["STRATEGY_PROFILE_MISSING"],
                            },
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        self.assertEqual(summary.status, STATUS_PASS)
        self.assertEqual(summary.layers["short_term"], "PASS")
        self.assertEqual(payload["metrics"]["order_intents"]["memory_blockers"], 0)
        self.assertGreater(payload["metrics"]["order_intents"]["advisory_memory_blockers"], 0)

    def test_no_live_ready_memory_blocker_still_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                "status": "DRY_RUN_PASSED",
                                "intent": {"pair": "EUR_USD"},
                                "risk_issues": [],
                                "strategy_issues": [],
                                "live_blockers": ["TELEMETRY_FORECAST_CONTEXT_REQUIRED_FOR_LIVE"],
                            }
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(summary.layers["short_term"], "BLOCK")
        self.assertTrue(any(issue["code"] == "SHORT_ORDER_INTENTS_MEMORY_BLOCKERS" for issue in payload["issues"]))


_NOW = datetime(2026, 6, 5, 0, 0, tzinfo=timezone.utc)


def _run(files: dict[str, Path]):
    return MemoryHealthAuditor(
        output_path=files["output"],
        report_path=files["report"],
    ).run(
        snapshot_path=files["snapshot"],
        target_state_path=files["target"],
        order_intents_path=files["intents"],
        strategy_profile_path=files["strategy_profile"],
        forecast_history_path=files["forecast_history"],
        projection_ledger_path=files["projection_ledger"],
        learning_audit_path=files["learning_audit"],
        entry_thesis_ledger_path=files["entry_thesis"],
        execution_ledger_db_path=files["execution_ledger"],
        now=_NOW,
    )


def _fixtures(
    root: Path,
    *,
    write_entry_thesis: bool = True,
    projection_expired: bool = False,
) -> dict[str, Path]:
    files = {
        "snapshot": root / "broker_snapshot.json",
        "target": root / "daily_target_state.json",
        "intents": root / "order_intents.json",
        "strategy_profile": root / "strategy_profile.json",
        "history_db": root / "legacy_history.db",
        "forecast_history": root / "forecast_history.jsonl",
        "projection_ledger": root / "projection_ledger.jsonl",
        "learning_audit": root / "learning_audit.json",
        "entry_thesis": root / "entry_thesis_ledger.jsonl",
        "execution_ledger": root / "execution_ledger.db",
        "output": root / "memory_health.json",
        "report": root / "memory_health_report.md",
    }
    files["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": _NOW.isoformat(),
                "account": {"last_transaction_id": "100", "fetched_at_utc": _NOW.isoformat()},
                "positions": [
                    {
                        "trade_id": "T1",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "owner": "trader",
                        "entry_price": 1.17,
                        "take_profit": 1.18,
                    }
                ],
                "orders": [],
                "quotes": {"EUR_USD": {"bid": 1.1701, "ask": 1.1702, "timestamp_utc": _NOW.isoformat()}},
            }
        )
    )
    files["target"].write_text(
        json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0})
    )
    files["intents"].write_text(
        json.dumps(
            {
                "results": [
                    {
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        "status": "LIVE_READY",
                        "intent": {"pair": "EUR_USD"},
                        "risk_issues": [],
                        "strategy_issues": [],
                        "live_blockers": [],
                    }
                ]
            }
        )
    )
    files["history_db"].write_text("mtime anchor")
    files["strategy_profile"].write_text(
        json.dumps(
            {
                "generated_at_utc": _NOW.isoformat(),
                "history_db": str(files["history_db"]),
                "profiles": [{"pair": "EUR_USD", "direction": "LONG", "method": "TREND_CONTINUATION"}],
            }
        )
    )
    files["forecast_history"].write_text(
        json.dumps(
            {
                "timestamp_utc": _NOW.isoformat(),
                "cycle_id": "cycle-1",
                "pair": "EUR_USD",
                "direction": "UP",
                "confidence": 0.8,
            }
        )
        + "\n"
    )
    emitted_at = _NOW - timedelta(minutes=90 if projection_expired else 1)
    files["projection_ledger"].write_text(
        json.dumps(
            {
                "timestamp_emitted_utc": emitted_at.isoformat(),
                "pair": "EUR_USD",
                "signal_name": "directional_forecast",
                "direction": "UP",
                "entry_price": 1.17,
                "predicted_target_price": 1.18,
                "resolution_window_min": 30.0,
                "resolution_status": "PENDING" if projection_expired else "HIT",
                "cycle_id": "cycle-1",
            }
        )
        + "\n"
    )
    files["learning_audit"].write_text(
        json.dumps(
            {
                "status": "LEARNING_AUDIT_PASS",
                "blockers": [],
                "warnings": [],
                "learning_influence": {"influenced_lanes": 0, "total_learning_score_delta": 0.0},
            }
        )
    )
    if write_entry_thesis:
        files["entry_thesis"].write_text(
            json.dumps(
                {
                    "timestamp_utc": _NOW.isoformat(),
                    "trade_id": "T1",
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "entry_price": 1.17,
                    "forecast_direction": "UP",
                    "forecast_confidence": 0.8,
                }
            )
            + "\n"
        )
    else:
        files["entry_thesis"].write_text("")
    _write_execution_ledger(files["execution_ledger"], last_transaction_id="100")
    return files


def _write_forecast_history(path: Path, timestamp: datetime) -> None:
    path.write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp.isoformat(),
                "cycle_id": "cycle-1",
                "pair": "EUR_USD",
                "direction": "UP",
                "confidence": 0.8,
            }
        )
        + "\n"
    )


def _write_execution_ledger(path: Path, *, last_transaction_id: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE oanda_transactions (transaction_id TEXT PRIMARY KEY);
            CREATE TABLE execution_events (event_uid TEXT PRIMARY KEY);
            """
        )
        conn.execute(
            "insert into sync_state(key, value, updated_at_utc) values (?, ?, ?)",
            ("last_oanda_transaction_id", last_transaction_id, _NOW.isoformat()),
        )
        conn.execute("insert into oanda_transactions(transaction_id) values (?)", ("99",))
        conn.execute("insert into execution_events(event_uid) values (?)", ("evt-1",))
