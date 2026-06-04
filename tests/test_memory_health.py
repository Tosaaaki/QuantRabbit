from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.memory_health import MemoryHealthAuditor, STATUS_BLOCKED, STATUS_PASS


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
