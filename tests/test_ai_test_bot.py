from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.ai_test_bot import AITestBotBacktester


class AITestBotBacktesterTest(unittest.TestCase):
    def test_defaults_use_recent_pretrade_evidence_with_high_support_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            rows = []
            for index in range(12):
                day = f"2026-04-{1 + (index % 5):02d}"
                rows.append(
                    (
                        "pretrade_outcomes",
                        day,
                        "EUR_USD",
                        "LONG",
                        100.0,
                        json.dumps({"id": f"train-{index}", "pretrade_level": "HIGH"}),
                    )
                )
            rows.extend(
                [
                    (
                        "pretrade_outcomes",
                        "2026-04-06",
                        "EUR_USD",
                        "LONG",
                        150.0,
                        json.dumps({"id": "validation-1", "pretrade_level": "HIGH"}),
                    ),
                    (
                        "pretrade_outcomes",
                        "2026-04-06",
                        "EUR_USD",
                        "LONG",
                        200.0,
                        json.dumps({"id": "validation-2", "pretrade_level": "HIGH"}),
                    ),
                ]
            )
            _seed_db(db, rows)

            summary = AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
            ).run(start_balance_jpy=10_000.0)

            self.assertEqual(summary.validation_days, 1)
            payload = json.loads((root / "ai_backtest.json").read_text())
            self.assertEqual(payload["source_tables"], ["trades", "pretrade_outcomes"])
            self.assertEqual(payload["training_days"], 5)
            self.assertEqual(payload["min_train_trades"], 12)
            day = payload["days"][0]
            self.assertEqual(
                day["selected_buckets"],
                ["pretrade_outcomes:EUR_USD:LONG:HIGH:UNSPECIFIED"],
            )
            self.assertEqual(day["managed_net_jpy"], 350.0)

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
            self.assertEqual(payload["firepower"]["best_selected_day_jpy"], 90.0)
            self.assertEqual(payload["bucket_contributions"][0]["bucket"], "trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED")
            self.assertEqual(payload["oracle"]["top_n_target_hit_days"], 1)
            self.assertEqual(payload["missed_best_days"][0]["best_bucket"], "trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED")
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
            self.assertEqual(payload["bucket_contributions"][0]["managed_net_jpy"], -75.0)

    def test_dedupes_seat_outcomes_by_overlapping_matched_trades_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            raw_train = json.dumps(
                {
                    "source": "s_hunt",
                    "setup_type": "MARKET",
                    "matched_trade_ids": "seat-1",
                    "updated_at": "2026-04-01 01:00:00",
                }
            )
            raw_train_later = json.dumps(
                {
                    "source": "s_hunt",
                    "setup_type": "MARKET",
                    "matched_trade_ids": "seat-1",
                    "updated_at": "2026-04-01 02:00:00",
                }
            )
            raw_train_overlap = json.dumps(
                {
                    "source": "s_hunt",
                    "setup_type": "MARKET",
                    "matched_trade_ids": "seat-1,seat-extra",
                    "updated_at": "2026-04-01 01:30:00",
                }
            )
            raw_validation = json.dumps(
                {
                    "source": "s_hunt",
                    "setup_type": "MARKET",
                    "matched_trade_ids": "seat-2",
                    "updated_at": "2026-04-02 01:00:00",
                }
            )
            raw_validation_overlap = json.dumps(
                {
                    "source": "s_hunt",
                    "setup_type": "MARKET",
                    "matched_trade_ids": "seat-2,seat-next",
                    "updated_at": "2026-04-02 02:00:00",
                }
            )
            _seed_db(
                db,
                [
                    ("seat_outcomes", "2026-04-01", "EUR_USD", "SHORT", 500.0, raw_train),
                    ("seat_outcomes", "2026-04-01", "EUR_USD", "SHORT", 500.0, raw_train_overlap),
                    ("seat_outcomes", "2026-04-01", "EUR_USD", "SHORT", 500.0, raw_train_later),
                    ("seat_outcomes", "2026-04-02", "EUR_USD", "SHORT", 700.0, raw_validation),
                    ("seat_outcomes", "2026-04-02", "EUR_USD", "SHORT", 700.0, raw_validation_overlap),
                ],
            )

            AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("seat_outcomes",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            self.assertEqual(payload["raw_rows"], 5)
            self.assertEqual(payload["deduped_rows"], 2)
            self.assertEqual(payload["deduped_away_rows"], 3)
            day = payload["days"][0]
            self.assertEqual(day["selected_trades"], 1)
            self.assertEqual(day["managed_net_jpy"], 700.0)
            self.assertEqual(day["selected_buckets"], ["seat_outcomes:EUR_USD:SHORT:MARKET:S_HUNT"])


def _seed_db(path: Path, rows: list[tuple[str, str, str, str, float] | tuple[str, str, str, str, float, str]]) -> None:
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
        expanded = []
        for row in rows:
            source, session_date, pair, direction, pl = row[:5]
            raw_json = row[5] if len(row) == 6 else "{}"
            expanded.append((source, session_date, pair, direction, pl, raw_json))
        conn.executemany(
            """
            INSERT INTO legacy_records(
                source_table, session_date, pair, direction, pl, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            expanded,
        )


if __name__ == "__main__":
    unittest.main()
