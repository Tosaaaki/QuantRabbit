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
    STATUS_ACTION_REQUIRED,
    STATUS_BLOCKED,
    SelfImprovementAuditor,
    _projection_expired,
)


class SelfImprovementAuditorTest(unittest.TestCase):
    def test_projection_expiry_uses_live_telemetry_grace(self) -> None:
        row = {
            "timestamp_emitted_utc": (_NOW - timedelta(minutes=32)).isoformat(),
            "resolution_window_min": 30.0,
        }
        self.assertFalse(_projection_expired(row, now=_NOW))

        row["timestamp_emitted_utc"] = (_NOW - timedelta(minutes=36)).isoformat()
        self.assertTrue(_projection_expired(row, now=_NOW))

    def test_blocks_missing_memory_projection_and_entry_thesis_holes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True, write_memory=False, projection_expired=True)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

            codes = {item["code"] for item in payload["findings"]}
            self.assertEqual(summary.status, STATUS_BLOCKED)
            self.assertGreaterEqual(summary.p0_findings, 3)
            self.assertIn("MEMORY_HEALTH_UNREADABLE", codes)
            self.assertIn("PROJECTION_LEDGER_EXPIRED_PENDING", codes)
            self.assertIn("ENTRY_THESIS_MISSING_FOR_OPEN_TRADES", codes)
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]
                finding_count = conn.execute("SELECT COUNT(*) FROM self_improvement_findings").fetchone()[0]
            self.assertEqual(run_count, 1)
            self.assertEqual(finding_count, summary.findings)

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
                        "--memory-health",
                        str(files["memory"]),
                        "--learning-audit",
                        str(files["learning"]),
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
                    ]
                )

            result = json.loads(stdout.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], STATUS_BLOCKED)
            self.assertTrue(files["output"].exists())
            self.assertTrue(files["report"].exists())


_NOW = datetime(2026, 6, 5, 0, 0, tzinfo=timezone.utc)


def _run(files: dict[str, Path]):
    return SelfImprovementAuditor(
        db_path=files["execution_db"],
        history_db_path=files["history_db"],
        output_path=files["output"],
        report_path=files["report"],
    ).run(
        snapshot_path=files["snapshot"],
        target_state_path=files["target"],
        order_intents_path=files["intents"],
        memory_health_path=files["memory"],
        learning_audit_path=files["learning"],
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
        now=_NOW,
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
) -> dict[str, Path]:
    files = {
        "execution_db": root / "execution_ledger.db",
        "history_db": root / "self_improvement_history.db",
        "output": root / "self_improvement.json",
        "report": root / "self_improvement.md",
        "snapshot": root / "broker_snapshot.json",
        "target": root / "daily_target_state.json",
        "intents": root / "order_intents.json",
        "memory": root / "memory_health.json",
        "learning": root / "learning_audit.json",
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
                "orders": [],
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
            json.dumps({"status": "MEMORY_HEALTH_PASS", "issues": [], "blockers": [], "warnings": []})
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
    files["verification"].write_text(
        json.dumps({"status": "OK", "blocking_observations": 0, "blocking_evidence": []})
    )
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
