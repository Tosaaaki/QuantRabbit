from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.ai_test_bot import AITestBotBacktester


class AITestBotBacktesterTest(unittest.TestCase):
    def test_walk_forward_policy_does_not_select_validation_only_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-04-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-04-02", "EUR_USD", "LONG", 120.0),
                    ("trades", "2026-04-03", "EUR_USD", "LONG", 90.0),
                    ("trades", "2026-04-03", "GBP_USD", "LONG", 5000.0),
                ],
            )

            summary = AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=2,
                min_train_trades=2,
                max_active_buckets=1,
            ).run(start_balance_jpy=500.0)

            self.assertEqual(summary.status, "TARGET_COVERAGE_CERTIFIED")
            self.assertEqual(summary.validation_days, 1)
            self.assertEqual(summary.total_managed_net_jpy, 90.0)
            payload = json.loads((root / "ai_backtest.json").read_text())
            day = payload["days"][0]
            self.assertEqual(day["selected_buckets"], ["trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED"])
            self.assertEqual(day["selected_trades"], 1)
            self.assertEqual(day["managed_net_jpy"], 90.0)
            self.assertNotIn("GBP_USD", " ".join(day["selected_buckets"]))
            self.assertFalse(payload["live_permission"])
            self.assertIn("offline research bot", (root / "ai_backtest.md").read_text())

    def test_applies_equity_loss_cap_to_validation_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-04-01", "EUR_USD", "SHORT", 200.0),
                    ("trades", "2026-04-02", "EUR_USD", "SHORT", -400.0),
                ],
            )

            AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=75.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            day = payload["days"][0]
            self.assertEqual(day["raw_net_jpy"], -400.0)
            self.assertEqual(day["managed_net_jpy"], -75.0)
            self.assertEqual(payload["status"], "BLOCKED")
            self.assertTrue(any("out-of-sample managed net is not positive" in item for item in payload["blockers"]))


def _seed_db(path: Path, rows: list[tuple[str, str, str, str, float]]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE legacy_records (
                source_table TEXT NOT NULL,
                source_id TEXT,
                session_date TEXT,
                pair TEXT,
                direction TEXT,
                pl REAL,
                execution_style TEXT,
                allocation_band TEXT,
                thesis TEXT,
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO legacy_records(
                source_table, session_date, pair, direction, pl, raw_json
            )
            VALUES (?, ?, ?, ?, ?, '{}')
            """,
            rows,
        )


if __name__ == "__main__":
    unittest.main()
