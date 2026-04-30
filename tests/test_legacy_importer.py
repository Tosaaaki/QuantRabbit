from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.legacy.importer import LegacyImporter


class LegacyImporterTest(unittest.TestCase):
    def test_imports_structured_rows_and_live_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "legacy"
            memory = archive / "collab_trade/memory"
            logs = archive / "logs"
            memory.mkdir(parents=True)
            logs.mkdir(parents=True)
            (archive / "docs").mkdir()
            (archive / "docs/SKILL_trader.md").write_text("# Trader\nold prompt\n")

            db = sqlite3.connect(memory / "memory.db")
            db.executescript(
                """
                CREATE TABLE trades (id INTEGER, session_date TEXT, pair TEXT, direction TEXT, pl REAL, thesis TEXT);
                CREATE TABLE pretrade_outcomes (
                    id INTEGER, session_date TEXT, pair TEXT, direction TEXT, pl REAL,
                    execution_style TEXT, allocation_band TEXT, thesis TEXT
                );
                CREATE TABLE seat_outcomes (id INTEGER, session_date TEXT, pair TEXT, direction TEXT, realized_pl REAL, why TEXT);
                CREATE TABLE chunks (id INTEGER, text TEXT);
                CREATE TABLE user_calls (id INTEGER, text TEXT);
                CREATE TABLE market_events (id INTEGER, text TEXT);
                INSERT INTO trades VALUES (1, '2026-04-30', 'EUR_USD', 'LONG', 120.0, 'test');
                INSERT INTO pretrade_outcomes VALUES (2, '2026-04-30', 'EUR_USD', 'LONG', 120.0, 'MARKET', 'B+', 'test');
                INSERT INTO seat_outcomes VALUES (3, '2026-04-30', 'EUR_USD', 'LONG', 120.0, 'seat');
                INSERT INTO chunks VALUES (4, 'lesson');
                INSERT INTO user_calls VALUES (5, 'call');
                INSERT INTO market_events VALUES (6, 'event');
                """
            )
            db.commit()
            db.close()

            (logs / "live_trade_log.txt").write_text(
                "[2026-04-30 15:16:39 UTC] ENTRY EUR_USD LONG 3000u @1.17326 id=470016 TP=1.17554 SL=1.17234 Sp=0.8pip | thesis=x\n"
                "[2026-04-30 15:20:00 UTC] CLOSE EUR_USD LONG 3000u @1.17554 P/L=+1000.0000JPY Sp=0.8pip reason=TAKE_PROFIT_ORDER id=470016 txn=470020\n"
            )
            (logs / "trader_journal.jsonl").write_text(
                json.dumps({"event": "order_sent", "pair": "EUR_USD", "direction": "LONG"}) + "\n"
            )

            out_db = root / "history.db"
            report = root / "report.md"
            summary = LegacyImporter(archive, out_db, report).run()

            self.assertEqual(summary.legacy_rows["trades"], 1)
            self.assertEqual(summary.legacy_rows["pretrade_outcomes"], 1)
            self.assertEqual(summary.live_trade_events, 2)
            self.assertEqual(summary.journal_events, 1)
            self.assertTrue(report.exists())

            conn = sqlite3.connect(out_db)
            try:
                live_count = conn.execute("SELECT COUNT(*) FROM live_trade_events").fetchone()[0]
                source_count = conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(live_count, 2)
            self.assertGreaterEqual(source_count, 3)

    def test_classifies_unbracketed_legacy_lines_without_fake_trade_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "legacy"
            logs = archive / "logs"
            logs.mkdir(parents=True)
            (logs / "live_trade_log.txt").write_text(
                "=== LOOP END | Session P/L: -437.2 JPY | Trades: 7 ===\n"
                "GBP_USD LONG 1500u @1.33565 | UPL=+243 JPY | TP=1.33979 SL=1.33279 | M5 H20\n"
                "[06:20Z CLOSE GBP_USD LONG 2000u @1.34154 PL=-626JPY reason=H1_flip Sp=1.3pip]\n"
                "Today: 49 fills W=19/L=30 WR=38.8% PL=-4294.82JPY (bot legacy trades included).\n"
            )

            out_db = root / "history.db"
            LegacyImporter(archive, out_db, root / "report.md").run()

            conn = sqlite3.connect(out_db)
            try:
                rows = conn.execute(
                    "SELECT action, pair, direction, units, pl_jpy FROM live_trade_events ORDER BY line_no"
                ).fetchall()
            finally:
                conn.close()

            self.assertEqual(rows[0], ("SESSION_SUMMARY", None, None, None, None))
            self.assertEqual(rows[1], ("POSITION_SNAPSHOT", "GBP_USD", "LONG", 1500, None))
            self.assertEqual(rows[2], ("CLOSE", "GBP_USD", "LONG", 2000, -626.0))
            self.assertEqual(rows[3], ("NOTE", None, None, None, None))


if __name__ == "__main__":
    unittest.main()
