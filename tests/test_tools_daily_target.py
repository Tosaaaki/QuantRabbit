from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


def _load_daily_target_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "daily_target.py"
    spec = importlib.util.spec_from_file_location("tools_daily_target", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


daily_target = _load_daily_target_module()


class DailyTargetToolTest(unittest.TestCase):
    def test_uses_utc_day_boundary_and_protects_after_five_without_extension_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_path = root / "snapshot.json"
            ledger_path = root / "ledger.db"
            day_start_dir = root / "day_start_nav"
            day_start_dir.mkdir()
            (day_start_dir / "2026-06-28.json").write_text(
                json.dumps({"day_start_nav": 100_000, "source": "test"})
            )
            snapshot_path.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": "2026-06-28T12:00:00+00:00",
                        "account": {
                            "nav_jpy": 105_000,
                            "unrealized_pl_jpy": 1_000,
                            "margin_used_jpy": 25_000,
                        },
                    }
                )
            )
            with sqlite3.connect(ledger_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        realized_pl_jpy REAL
                    )
                    """
                )
                conn.executemany(
                    "INSERT INTO execution_events VALUES (?, ?, ?)",
                    [
                        ("2026-06-27T23:55:00+00:00", "TRADE_CLOSED", 999.0),
                        ("2026-06-28T00:15:00+00:00", "TRADE_CLOSED", 2_000.0),
                        ("2026-06-28T23:59:59+00:00", "TRADE_REDUCED", -500.0),
                        ("2026-06-29T00:00:00+00:00", "TRADE_CLOSED", 777.0),
                    ],
                )

            metrics = daily_target.compute_daily_target(
                snapshot_path=snapshot_path,
                execution_ledger_db=ledger_path,
                day_start_dir=day_start_dir,
                now_utc=datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc),
                dry_run=True,
                extension_gate=False,
            )

            self.assertEqual(metrics.trading_day_utc, "2026-06-28")
            self.assertEqual(metrics.realized_pl_today, 1_500.0)
            self.assertEqual(metrics.total_day_progress_yen, 5_000.0)
            self.assertEqual(metrics.total_day_progress_pct, 5.0)
            self.assertEqual(metrics.remaining_to_5pct_yen, 0.0)
            self.assertEqual(metrics.remaining_to_10pct_yen, 5_000.0)
            self.assertEqual(metrics.mode, "PROTECT")

    def test_extension_gate_is_explicit_after_base_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_path = root / "snapshot.json"
            ledger_path = root / "ledger.db"
            day_start_dir = root / "day_start_nav"
            day_start_dir.mkdir()
            (day_start_dir / "2026-06-28.json").write_text(
                json.dumps({"day_start_nav": 100_000, "source": "test"})
            )
            snapshot_path.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": "2026-06-28T12:00:00+00:00",
                        "account": {"nav_jpy": 106_000, "unrealized_pl_jpy": 0, "margin_used_jpy": 0},
                    }
                )
            )
            with sqlite3.connect(ledger_path) as conn:
                conn.execute("CREATE TABLE execution_events (ts_utc TEXT, event_type TEXT, realized_pl_jpy REAL)")

            metrics = daily_target.compute_daily_target(
                snapshot_path=snapshot_path,
                execution_ledger_db=ledger_path,
                day_start_dir=day_start_dir,
                now_utc=datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc),
                dry_run=True,
                extension_gate=True,
            )

            self.assertEqual(metrics.mode, "EXTEND")

    def test_persists_day_start_nav_when_missing_and_not_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_path = root / "snapshot.json"
            ledger_path = root / "ledger.db"
            day_start_dir = root / "day_start_nav"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": "2026-06-28T00:03:00+00:00",
                        "account": {
                            "nav_jpy": 123_456.7,
                            "unrealized_pl_jpy": 0,
                            "margin_used_jpy": 0,
                            "last_transaction_id": "42",
                        },
                    }
                )
            )
            with sqlite3.connect(ledger_path) as conn:
                conn.execute("CREATE TABLE execution_events (ts_utc TEXT, event_type TEXT, realized_pl_jpy REAL)")

            metrics = daily_target.compute_daily_target(
                snapshot_path=snapshot_path,
                execution_ledger_db=ledger_path,
                day_start_dir=day_start_dir,
                now_utc=datetime(2026, 6, 28, 0, 3, tzinfo=timezone.utc),
                dry_run=False,
            )

            record_path = day_start_dir / "2026-06-28.json"
            self.assertTrue(record_path.exists())
            record = json.loads(record_path.read_text())
            self.assertEqual(record["day_start_nav"], 123_456.7)
            self.assertEqual(record["trading_day_utc"], "2026-06-28")
            self.assertEqual(metrics.day_start_nav_path, str(record_path))
