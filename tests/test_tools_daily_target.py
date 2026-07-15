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


def _load_session_data_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "session_data.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("tools_session_data", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass


daily_target = _load_daily_target_module()
session_data = _load_session_data_module()


class DailyTargetToolTest(unittest.TestCase):
    def test_session_data_help_renders_literal_percent(self) -> None:
        help_text = session_data._build_parser().format_help()

        self.assertIn("+10% mode", " ".join(help_text.split()))

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
            self.assertEqual(metrics.rolling_30d_policy, "ROLLING_30D_4X")
            self.assertEqual(metrics.rolling_30d_start_equity, 105_000.0)
            self.assertEqual(metrics.current_equity_raw, 105_000.0)
            self.assertEqual(metrics.capital_flows_30d, 0.0)
            self.assertEqual(metrics.funding_adjusted_equity, 105_000.0)
            self.assertEqual(metrics.current_equity, 105_000.0)
            self.assertEqual(metrics.rolling_30d_multiplier_raw, 1.0)
            self.assertEqual(metrics.rolling_30d_multiplier_funding_adjusted, 1.0)
            self.assertEqual(metrics.current_30d_multiplier, 1.0)
            self.assertEqual(metrics.remaining_to_4x_raw, 315_000.0)
            self.assertEqual(metrics.remaining_to_4x_funding_adjusted, 315_000.0)
            self.assertEqual(metrics.remaining_to_4x, 315_000.0)
            self.assertEqual(metrics.required_calendar_daily_return_raw, metrics.required_calendar_daily_return)
            self.assertEqual(metrics.required_active_day_return_raw, metrics.required_active_day_return)
            self.assertEqual(
                metrics.required_calendar_daily_return_funding_adjusted,
                metrics.required_calendar_daily_return,
            )
            self.assertEqual(
                metrics.required_active_day_return_funding_adjusted,
                metrics.required_active_day_return,
            )
            self.assertEqual(metrics.performance_basis, "funding_adjusted")
            self.assertEqual(metrics.sizing_basis, "raw_nav")
            self.assertIsNotNone(metrics.required_calendar_daily_return)
            self.assertIsNotNone(metrics.required_active_day_return)
            self.assertGreater(
                metrics.required_active_day_return or 0.0,
                metrics.required_calendar_daily_return or 0.0,
            )
            self.assertEqual(metrics.pace_state, "AHEAD")
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

    def test_capital_flow_adjusts_rolling_multiplier_without_changing_raw_nav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_path = root / "snapshot.json"
            ledger_path = root / "ledger.db"
            day_start_dir = root / "day_start_nav"
            capital_flows_path = root / "capital_flows.json"
            day_start_dir.mkdir()
            (day_start_dir / "2026-07-02.json").write_text(
                json.dumps({"day_start_nav": 300_000, "source": "test"})
            )
            (day_start_dir / "rolling_30d_4x.json").write_text(
                json.dumps(
                    {
                        "rolling_30d_start_utc": "2026-07-01T00:00:00+00:00",
                        "rolling_30d_start_equity": 200_000.0,
                    }
                )
            )
            capital_flows_path.write_text(
                json.dumps(
                    {
                        "capital_flows": [
                            {
                                "timestamp_utc": "2026-07-01T12:00:00Z",
                                "amount_jpy": 100_000,
                                "type": "DEPOSIT",
                                "source": "operator",
                                "note": "100,000 JPY capital injection",
                                "included_in_raw_equity": True,
                                "excluded_from_funding_adjusted_return": True,
                            }
                        ]
                    }
                )
            )
            snapshot_path.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": "2026-07-02T00:00:00+00:00",
                        "account": {"nav_jpy": 300_000, "unrealized_pl_jpy": 0, "margin_used_jpy": 30_000},
                    }
                )
            )
            with sqlite3.connect(ledger_path) as conn:
                conn.execute("CREATE TABLE execution_events (ts_utc TEXT, event_type TEXT, realized_pl_jpy REAL)")

            metrics = daily_target.compute_daily_target(
                snapshot_path=snapshot_path,
                execution_ledger_db=ledger_path,
                day_start_dir=day_start_dir,
                capital_flows_path=capital_flows_path,
                now_utc=datetime(2026, 7, 2, 0, 0, tzinfo=timezone.utc),
                dry_run=True,
            )
            report = daily_target.format_daily_target_block(metrics)

            self.assertEqual(metrics.current_nav, 300_000.0)
            self.assertEqual(metrics.current_equity_raw, 300_000.0)
            self.assertEqual(metrics.capital_flows_30d, 100_000.0)
            self.assertEqual(metrics.funding_adjusted_equity, 200_000.0)
            self.assertEqual(metrics.rolling_30d_multiplier_raw, 1.5)
            self.assertEqual(metrics.rolling_30d_multiplier_funding_adjusted, 1.0)
            self.assertEqual(metrics.current_30d_multiplier, 1.0)
            self.assertEqual(metrics.remaining_to_4x_raw, 500_000.0)
            self.assertEqual(metrics.remaining_to_4x_funding_adjusted, 600_000.0)
            self.assertLess(
                metrics.required_calendar_daily_return_raw or 0.0,
                metrics.required_calendar_daily_return_funding_adjusted or 0.0,
            )
            self.assertEqual(metrics.required_calendar_daily_return, metrics.required_calendar_daily_return_funding_adjusted)
            self.assertEqual(metrics.required_active_day_return, metrics.required_active_day_return_funding_adjusted)
            self.assertEqual(metrics.performance_basis, "funding_adjusted")
            self.assertEqual(metrics.sizing_basis, "raw_nav")
            self.assertIn("current_equity_raw", report)
            self.assertIn("funding_adjusted_equity", report)
            self.assertIn("rolling_30d_multiplier_raw", report)
            self.assertIn("rolling_30d_multiplier_funding_adjusted", report)
            self.assertIn("required_calendar_daily_return_funding_adjusted", report)
            self.assertIn("performance_basis", report)

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

    def test_full_trader_board_requires_hero_path_attack_stack_and_blocker(self) -> None:
        metrics = daily_target.DailyTargetMetrics(
            trading_day_utc="2026-06-28",
            day_start_utc="2026-06-28T00:00:00+00:00",
            day_end_utc="2026-06-29T00:00:00+00:00",
            generated_at_utc="2026-06-28T12:00:00+00:00",
            snapshot_fetched_at_utc="2026-06-28T12:00:00+00:00",
            day_start_nav=100_000.0,
            current_nav=101_000.0,
            realized_pl_today=1_000.0,
            unrealized_pl=0.0,
            total_day_progress_yen=1_000.0,
            total_day_progress_pct=1.0,
            rolling_30d_policy="ROLLING_30D_4X",
            rolling_30d_start_utc="2026-06-28T12:00:00+00:00",
            rolling_30d_end_utc="2026-07-28T12:00:00+00:00",
            rolling_30d_start_equity=100_000.0,
            current_equity_raw=101_000.0,
            capital_flows_30d=0.0,
            capital_flow_count_30d=0,
            funding_adjusted_equity=101_000.0,
            current_equity=101_000.0,
            rolling_30d_multiplier_raw=1.01,
            rolling_30d_multiplier_funding_adjusted=1.01,
            current_30d_multiplier=1.01,
            remaining_to_4x_raw=299_000.0,
            remaining_to_4x_funding_adjusted=299_000.0,
            remaining_to_4x=299_000.0,
            required_calendar_daily_return_raw=4.686,
            required_active_day_return_raw=6.534,
            required_calendar_daily_return_funding_adjusted=4.686,
            required_active_day_return_funding_adjusted=6.534,
            required_calendar_daily_return=4.686,
            required_active_day_return=6.534,
            performance_basis="funding_adjusted",
            sizing_basis="raw_nav",
            pace_state="AHEAD",
            capital_flow_issues=(),
            base_target_yen=5_000.0,
            extension_target_yen=10_000.0,
            remaining_to_5pct_yen=4_000.0,
            remaining_to_10pct_yen=9_000.0,
            margin_used=0.0,
            margin_pct=0.0,
            mode="BUILD",
            extension_gate=False,
            day_start_nav_source="test",
            day_start_nav_path="test",
        )

        board = session_data.format_full_trader_board(metrics)

        self.assertIn("## 5% PACE BOARD", board)
        self.assertIn("Remaining to +5%: 4,000 JPY", board)
        self.assertIn("not forced churn", board)
        self.assertIn("Path A / HERO:", board)
        self.assertIn("Path B / SECOND SHOT:", board)
        self.assertIn("Path C / NO HONEST PATH:", board)
        self.assertIn("## ATTACK STACK", board)
        self.assertIn("Why this thesis can still reach +5% today:", board)
        self.assertIn("B/C trades cannot be the +5% pace path.", board)
        self.assertIn("One distant pending order is not enough.", board)
        self.assertIn("## 10% EXTENSION GATE", board)
        self.assertIn("EXTEND mode requires A/S grade risk.", board)
