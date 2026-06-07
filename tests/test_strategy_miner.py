from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.strategy.miner import StrategyMiner


class StrategyMinerTest(unittest.TestCase):
    def test_mines_candidates_blocks_and_missed_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "history.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE source_files (rel_path TEXT, kind TEXT, size_bytes INTEGER, sha256 TEXT, mtime_utc TEXT);
                CREATE TABLE legacy_records (
                    source_table TEXT, source_id TEXT, session_date TEXT, pair TEXT, direction TEXT,
                    pl REAL, execution_style TEXT, allocation_band TEXT, thesis TEXT, raw_json TEXT
                );
                CREATE TABLE live_trade_events (
                    line_no INTEGER, timestamp_text TEXT, action TEXT, pair TEXT, direction TEXT,
                    units INTEGER, price TEXT, pl_jpy REAL, spread_pips REAL, trade_id TEXT, reason TEXT, raw_line TEXT
                );
                CREATE TABLE jsonl_events (
                    source_name TEXT, line_no INTEGER, event_type TEXT, timestamp_utc TEXT,
                    pair TEXT, direction TEXT, raw_json TEXT
                );
                """
            )
            conn.executemany(
                """
                INSERT INTO legacy_records VALUES
                ('pretrade_outcomes', ?, '2026-04-30', 'EUR_USD', 'LONG', ?, 'MARKET', 'B+', 'edge', ?)
                """,
                [(str(i), 100.0, json.dumps({"id": i})) for i in range(6)],
            )
            conn.execute(
                "INSERT INTO live_trade_events VALUES (1, 't', 'CLOSE', 'EUR_USD', 'LONG', 1000, '1.0', 120.0, 0.8, '1', '', '')"
            )
            conn.executemany(
                """
                INSERT INTO legacy_records VALUES
                ('pretrade_outcomes', ?, '2026-04-30', 'GBP_USD', 'LONG', ?, 'MARKET', 'B+', 'edge', ?)
                """,
                [(f"g{i}", 100.0, json.dumps({"id": f"g{i}"})) for i in range(5)],
            )
            conn.execute(
                "INSERT INTO live_trade_events VALUES (4, 't', 'CLOSE', 'GBP_USD', 'LONG', 1000, '1.0', 1000.0, 0.8, '4', '', '')"
            )
            conn.execute(
                "INSERT INTO live_trade_events VALUES (5, 't', 'CLOSE', 'GBP_USD', 'LONG', 1000, '1.0', -600.0, 0.8, '5', '', '')"
            )
            conn.execute(
                "INSERT INTO live_trade_events VALUES (2, 't', 'CLOSE', 'USD_JPY', 'LONG', 20000, '157.0', -900.0, 0.8, '2', 'loss', '')"
            )
            for i in range(3):
                conn.execute(
                    "INSERT INTO legacy_records VALUES ('seat_outcomes', ?, '2026-04-30', 'AUD_USD', 'LONG', NULL, NULL, NULL, NULL, ?)",
                    (
                        str(i),
                        json.dumps(
                            {
                                "id": i,
                                "pair": "AUD_USD",
                                "direction": "LONG",
                                "discovered": 1,
                                "missed": 1,
                                "captured": 0,
                                "directionally_correct": 1,
                            }
                        ),
                    ),
                )
            conn.execute(
                "INSERT INTO jsonl_events VALUES ('trader_journal', 1, 'order_blocked', 't', 'USD_JPY', 'LONG', ?)",
                (json.dumps({"reason": "loss cap"}),),
            )
            conn.commit()
            conn.close()

            report = root / "strategy.md"
            profile = root / "profile.json"
            summary = StrategyMiner(db_path, report, profile, loss_cap_jpy=500).run()

            self.assertEqual(summary.candidates, 1)
            self.assertEqual(summary.risk_repair_candidates, 1)
            self.assertEqual(summary.blocked, 1)
            self.assertEqual(summary.mined_missed_edges, 1)
            payload = json.loads(profile.read_text())
            statuses = {f"{item['pair']} {item['direction']}": item["status"] for item in payload["profiles"]}
            profiles = {f"{item['pair']} {item['direction']}": item for item in payload["profiles"]}
            self.assertEqual(statuses["EUR_USD LONG"], "CANDIDATE")
            self.assertEqual(statuses["GBP_USD LONG"], "RISK_REPAIR_CANDIDATE")
            self.assertEqual(statuses["USD_JPY LONG"], "BLOCK_UNTIL_NEW_EVIDENCE")
            self.assertEqual(statuses["AUD_USD LONG"], "MINE_MISSED_EDGE")
            self.assertEqual(profiles["GBP_USD LONG"]["positive_best_jpy"], 1000.0)
            self.assertEqual(profiles["AUD_USD LONG"]["seat_pl_n"], 0)
            self.assertGreaterEqual(profiles["GBP_USD LONG"]["target_reward_risk"], 1.5)
            self.assertIn("Generated System Rules", report.read_text())

    def test_refuses_to_overwrite_with_empty_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "history.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE source_files (rel_path TEXT, kind TEXT, size_bytes INTEGER, sha256 TEXT, mtime_utc TEXT);
                CREATE TABLE legacy_records (
                    source_table TEXT, source_id TEXT, session_date TEXT, pair TEXT, direction TEXT,
                    pl REAL, execution_style TEXT, allocation_band TEXT, thesis TEXT, raw_json TEXT
                );
                CREATE TABLE live_trade_events (
                    line_no INTEGER, timestamp_text TEXT, action TEXT, pair TEXT, direction TEXT,
                    units INTEGER, price TEXT, pl_jpy REAL, spread_pips REAL, trade_id TEXT, reason TEXT, raw_line TEXT
                );
                CREATE TABLE jsonl_events (
                    source_name TEXT, line_no INTEGER, event_type TEXT, timestamp_utc TEXT,
                    pair TEXT, direction TEXT, raw_json TEXT
                );
                """
            )
            conn.commit()
            conn.close()
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "status": "CANDIDATE",
                            }
                        ]
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "zero profiles"):
                StrategyMiner(db_path, root / "strategy.md", profile, loss_cap_jpy=500).run()

            payload = json.loads(profile.read_text())
            self.assertEqual(len(payload["profiles"]), 1)
            self.assertEqual(payload["profiles"][0]["pair"], "EUR_USD")

    def test_preserves_method_receipt_promotion_on_remine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "history.db"
            _create_schema(db_path)
            with sqlite3.connect(db_path) as conn:
                for i in range(3):
                    conn.execute(
                        "INSERT INTO legacy_records VALUES ('seat_outcomes', ?, '2026-04-30', 'AUD_USD', 'LONG', NULL, NULL, NULL, NULL, ?)",
                        (
                            str(i),
                            json.dumps(
                                {
                                    "id": i,
                                    "pair": "AUD_USD",
                                    "direction": "LONG",
                                    "discovered": 1,
                                    "missed": 1,
                                    "captured": 0,
                                    "directionally_correct": 1,
                                }
                            ),
                        ),
                    )

            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "AUD_USD",
                                "direction": "LONG",
                                "method": "RANGE_ROTATION",
                                "status": "CANDIDATE",
                                "required_fix": "promoted from MINE_MISSED_EDGE by trigger receipt",
                                "receipt_promotion": {
                                    "from_status": "MINE_MISSED_EDGE",
                                    "lane_id": "range_trader:AUD_USD:LONG:RANGE_ROTATION",
                                    "method": "RANGE_ROTATION",
                                    "promoted_at_utc": "2026-06-05T00:00:00+00:00",
                                    "reason": "missed edge converted into LIMIT trigger receipt",
                                },
                            }
                        ]
                    }
                )
            )

            summary = StrategyMiner(db_path, root / "strategy.md", profile, loss_cap_jpy=500).run()

            self.assertEqual(summary.candidates, 1)
            payload = json.loads(profile.read_text())
            profiles = {
                (item["pair"], item["direction"], item.get("method")): item
                for item in payload["profiles"]
            }
            self.assertEqual(profiles[("AUD_USD", "LONG", None)]["status"], "MINE_MISSED_EDGE")
            self.assertEqual(profiles[("AUD_USD", "LONG", "RANGE_ROTATION")]["status"], "CANDIDATE")
            self.assertEqual(payload["last_receipt_promotion_at_utc"], "2026-06-05T00:00:00+00:00")

    def test_drops_missed_edge_promotion_when_seat_realized_net_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "history.db"
            _create_schema(db_path)
            with sqlite3.connect(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO legacy_records VALUES
                    ('pretrade_outcomes', ?, '2026-04-30', 'AUD_JPY', 'LONG', ?, 'MARKET', 'B+', 'edge', ?)
                    """,
                    [(str(i), 100.0, json.dumps({"id": i})) for i in range(5)],
                )
                conn.execute(
                    "INSERT INTO live_trade_events VALUES (1, 't', 'CLOSE', 'AUD_JPY', 'LONG', 1000, '100.0', 250.0, 0.8, '1', '', '')"
                )
                for i in range(3):
                    conn.execute(
                        "INSERT INTO legacy_records VALUES ('seat_outcomes', ?, '2026-04-30', 'AUD_JPY', 'LONG', ?, NULL, NULL, NULL, ?)",
                        (
                            f"seat-{i}",
                            -200.0,
                            json.dumps(
                                {
                                    "id": i,
                                    "pair": "AUD_JPY",
                                    "direction": "LONG",
                                    "discovered": 1,
                                    "missed": 1,
                                    "captured": 0,
                                    "directionally_correct": 1,
                                }
                            ),
                        ),
                    )

            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "AUD_JPY",
                                "direction": "LONG",
                                "method": "RANGE_ROTATION",
                                "status": "CANDIDATE",
                                "required_fix": "promoted from MINE_MISSED_EDGE by trigger receipt",
                                "receipt_promotion": {
                                    "from_status": "MINE_MISSED_EDGE",
                                    "lane_id": "range_trader:AUD_JPY:LONG:RANGE_ROTATION",
                                    "method": "RANGE_ROTATION",
                                    "promoted_at_utc": "2026-06-05T00:00:00+00:00",
                                    "reason": "missed edge converted into LIMIT trigger receipt",
                                },
                            }
                        ]
                    }
                )
            )

            summary = StrategyMiner(db_path, root / "strategy.md", profile, loss_cap_jpy=500).run()

            self.assertEqual(summary.candidates, 1)
            payload = json.loads(profile.read_text())
            profiles = {
                (item["pair"], item["direction"], item.get("method")): item
                for item in payload["profiles"]
            }
            self.assertEqual(profiles[("AUD_JPY", "LONG", None)]["status"], "CANDIDATE")
            self.assertEqual(profiles[("AUD_JPY", "LONG", None)]["seat_net_jpy"], -600.0)
            self.assertNotIn(("AUD_JPY", "LONG", "RANGE_ROTATION"), profiles)
            self.assertNotIn("last_receipt_promotion_at_utc", payload)

    def test_drops_receipt_promotion_when_current_history_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "history.db"
            _create_schema(db_path)
            with sqlite3.connect(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO legacy_records VALUES
                    ('pretrade_outcomes', ?, '2026-04-30', 'GBP_USD', 'LONG', ?, 'MARKET', 'B+', 'weak', ?)
                    """,
                    [(str(i), -100.0, json.dumps({"id": i})) for i in range(5)],
                )
                conn.executemany(
                    "INSERT INTO live_trade_events VALUES (?, 't', 'CLOSE', 'GBP_USD', 'LONG', 1000, '1.0', ?, 0.8, ?, '', '')",
                    [(i, -50.0, str(i)) for i in range(1, 4)],
                )

            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "GBP_USD",
                                "direction": "LONG",
                                "method": "BREAKOUT_FAILURE",
                                "status": "CANDIDATE",
                                "required_fix": "promoted from old receipt",
                                "receipt_promotion": {
                                    "from_status": "MINE_MISSED_EDGE",
                                    "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                                    "method": "BREAKOUT_FAILURE",
                                    "promoted_at_utc": "2026-06-05T00:00:00+00:00",
                                    "reason": "old trigger receipt",
                                },
                            }
                        ]
                    }
                )
            )

            summary = StrategyMiner(db_path, root / "strategy.md", profile, loss_cap_jpy=500).run()

            self.assertEqual(summary.candidates, 0)
            payload = json.loads(profile.read_text())
            profiles = {
                (item["pair"], item["direction"], item.get("method")): item
                for item in payload["profiles"]
            }
            self.assertEqual(profiles[("GBP_USD", "LONG", None)]["status"], "BLOCK_UNTIL_NEW_EVIDENCE")
            self.assertNotIn(("GBP_USD", "LONG", "BREAKOUT_FAILURE"), profiles)


def _create_schema(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE source_files (rel_path TEXT, kind TEXT, size_bytes INTEGER, sha256 TEXT, mtime_utc TEXT);
            CREATE TABLE legacy_records (
                source_table TEXT, source_id TEXT, session_date TEXT, pair TEXT, direction TEXT,
                pl REAL, execution_style TEXT, allocation_band TEXT, thesis TEXT, raw_json TEXT
            );
            CREATE TABLE live_trade_events (
                line_no INTEGER, timestamp_text TEXT, action TEXT, pair TEXT, direction TEXT,
                units INTEGER, price TEXT, pl_jpy REAL, spread_pips REAL, trade_id TEXT, reason TEXT, raw_line TEXT
            );
            CREATE TABLE jsonl_events (
                source_name TEXT, line_no INTEGER, event_type TEXT, timestamp_utc TEXT,
                pair TEXT, direction TEXT, raw_json TEXT
            );
            """
        )


if __name__ == "__main__":
    unittest.main()
