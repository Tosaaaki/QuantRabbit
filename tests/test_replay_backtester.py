from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.replay import ReplayBacktester


class ReplayBacktesterTest(unittest.TestCase):
    def test_replays_legacy_days_against_target_and_risk_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_db(db)

            summary = ReplayBacktester(
                db_path=db,
                output_path=root / "replay.json",
                report_path=root / "replay.md",
                max_loss_jpy=500,
            ).run(start_balance_jpy=10_000)

            self.assertEqual(summary.days, 2)
            self.assertEqual(summary.target_jpy, 1000)
            self.assertEqual(summary.historical_target_hits, 1)
            self.assertEqual(summary.evidence_target_covered, 2)
            self.assertEqual(summary.risk_repair_days, 1)
            payload = json.loads((root / "replay.json").read_text())
            days = {item["session_date"]: item for item in payload["days"]}
            self.assertEqual(days["2026-04-01"]["status"], "HISTORICAL_TARGET_HIT")
            self.assertEqual(days["2026-04-02"]["status"], "EVIDENCE_COVERS_TARGET")
            self.assertEqual(days["2026-04-02"]["risk_capped_net_jpy"], -300)
            self.assertIn("worst legacy loss -900 JPY requires risk repair", days["2026-04-02"]["blockers"])
            self.assertIn("Replay Contract", (root / "replay.md").read_text())

    def test_limits_to_most_recent_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_db(db)

            ReplayBacktester(db_path=db, output_path=root / "replay.json", report_path=root / "replay.md").run(
                start_balance_jpy=10_000,
                max_days=1,
            )

            payload = json.loads((root / "replay.json").read_text())
            self.assertEqual([item["session_date"] for item in payload["days"]], ["2026-04-02"])


def _seed_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
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
            );
            CREATE TABLE live_trade_events (
                line_no INTEGER PRIMARY KEY,
                timestamp_text TEXT,
                action TEXT,
                pair TEXT,
                direction TEXT,
                units INTEGER,
                price TEXT,
                pl_jpy REAL,
                spread_pips REAL,
                trade_id TEXT,
                reason TEXT,
                raw_line TEXT NOT NULL
            );
            """
        )
        rows = [
            ("trades", "2026-04-01", "EUR_USD", "LONG", 700.0),
            ("trades", "2026-04-01", "EUR_USD", "LONG", 500.0),
            ("pretrade_outcomes", "2026-04-01", "EUR_USD", "LONG", 300.0),
            ("seat_outcomes", "2026-04-01", "EUR_USD", "LONG", 200.0),
            ("trades", "2026-04-02", "GBP_USD", "LONG", -900.0),
            ("trades", "2026-04-02", "GBP_USD", "LONG", 200.0),
            ("pretrade_outcomes", "2026-04-02", "GBP_USD", "LONG", 1300.0),
            ("seat_outcomes", "2026-04-02", "GBP_USD", "LONG", 1600.0),
        ]
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
