from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.outcome_mart import OutcomeMartBuilder


class OutcomeMartBuilderTest(unittest.TestCase):
    def test_builds_method_edges_from_archive_outcomes_and_story_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_history_db(db)

            summary = OutcomeMartBuilder(
                db_path=db,
                execution_ledger_db_path=root / "missing_execution_ledger.db",
                output_path=root / "outcome_mart.json",
                report_path=root / "outcome_mart.md",
            ).run()

            payload = json.loads((root / "outcome_mart.json").read_text())
            self.assertEqual(summary.status, "OUTCOME_MART_READY")
            self.assertFalse(payload["live_permission"])
            self.assertTrue(payload["read_only"])
            self.assertEqual(payload["source_coverage"]["archive_outcomes"], 3)
            self.assertEqual(payload["source_coverage"]["story_observations"], 2)
            self.assertIn("condition_rollups", payload)
            self.assertIn("condition_validation", payload)
            conditions = {item["key"]: item for item in payload["condition_edges"]}
            condition_key = "ALL:ALL:TREND_CONTINUATION:MARKET:LONDON:TRENDING"
            self.assertEqual(conditions[condition_key]["outcome_n"], 2)
            self.assertEqual(conditions[condition_key]["net_jpy"], 40.0)
            self.assertEqual(conditions[condition_key]["evidence_state"], "POSITIVE_ARCHIVE_EDGE")
            edges = {item["key"]: item for item in payload["method_edges"]}
            eur_key = "EUR_USD:LONG:TREND_CONTINUATION:ALL:ALL:ALL"
            aud_key = "AUD_JPY:SHORT:RANGE_ROTATION:ALL:ALL:ALL"
            self.assertEqual(edges[eur_key]["outcome_n"], 2)
            self.assertEqual(edges[eur_key]["net_jpy"], 40.0)
            self.assertEqual(edges[eur_key]["evidence_state"], "POSITIVE_ARCHIVE_EDGE")
            self.assertEqual(edges[aud_key]["story_observation_n"], 1)
            report = (root / "outcome_mart.md").read_text()
            self.assertIn("Winning Conditions", report)
            self.assertIn("Walk-Forward Condition Validation", report)
            self.assertIn("Winning Condition Rollups", report)
            self.assertIn("Pair/Method Drilldown", report)

    def test_cli_builds_outcome_mart_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_history_db(db)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "build-outcome-mart",
                        "--db",
                        str(db),
                        "--execution-ledger-db",
                        str(root / "missing.db"),
                        "--output",
                        str(root / "outcome_mart.json"),
                        "--report",
                        str(root / "outcome_mart.md"),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "OUTCOME_MART_READY")
            self.assertEqual(payload["archive_outcomes"], 3)
            self.assertGreater(payload["condition_edges"], 0)
            self.assertGreater(payload["condition_rollups"], 0)
            self.assertIn("validated_condition_outcomes", payload)
            self.assertFalse(payload["live_permission"])

    def test_restores_archive_conditions_from_text_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_text_condition_history_db(db)

            OutcomeMartBuilder(
                db_path=db,
                execution_ledger_db_path=root / "missing_execution_ledger.db",
                output_path=root / "outcome_mart.json",
                report_path=root / "outcome_mart.md",
            ).run()

            payload = json.loads((root / "outcome_mart.json").read_text())
            conditions = {item["key"]: item for item in payload["condition_edges"]}
            self.assertIn("ALL:ALL:RANGE_ROTATION:LIMIT:NY:RANGE", conditions)
            condition = conditions["ALL:ALL:RANGE_ROTATION:LIMIT:NY:RANGE"]
            self.assertEqual(condition["outcome_n"], 1)
            self.assertEqual(condition["net_jpy"], 210.0)

    def test_walk_forward_validates_condition_edges_using_prior_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_walk_forward_history_db(db)

            summary = OutcomeMartBuilder(
                db_path=db,
                execution_ledger_db_path=root / "missing_execution_ledger.db",
                output_path=root / "outcome_mart.json",
                report_path=root / "outcome_mart.md",
            ).run()

            payload = json.loads((root / "outcome_mart.json").read_text())
            validation = payload["condition_validation"]
            self.assertEqual(validation["status"], "CONDITION_WALK_FORWARD_READY")
            self.assertEqual(validation["validated_outcomes"], 2)
            self.assertEqual(validation["directional_hit_outcomes"], 2)
            self.assertEqual(validation["directional_hit_rate_pct"], 100.0)
            self.assertEqual(validation["predicted_positive"]["outcomes"], 1)
            self.assertEqual(validation["predicted_negative"]["outcomes"], 1)
            matched_edges = {item["key"]: item for item in validation["matched_edges"]}
            self.assertEqual(
                matched_edges["ALL:ALL:TREND_CONTINUATION:MARKET:ASIA:TRENDING"]["predicted_edge"],
                "POSITIVE",
            )
            self.assertEqual(
                matched_edges["ALL:ALL:BREAKOUT_FAILURE:MARKET:ASIA:RANGE"]["predicted_edge"],
                "NEGATIVE",
            )
            self.assertEqual(summary.validated_condition_outcomes, 2)
            self.assertEqual(summary.condition_directional_hit_rate_pct, 100.0)

    def test_walk_forward_does_not_train_on_same_timestamp_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_same_timestamp_history_db(db)

            OutcomeMartBuilder(
                db_path=db,
                execution_ledger_db_path=root / "missing_execution_ledger.db",
                output_path=root / "outcome_mart.json",
                report_path=root / "outcome_mart.md",
            ).run()

            payload = json.loads((root / "outcome_mart.json").read_text())
            validation = payload["condition_validation"]
            self.assertEqual(validation["eligible_outcomes"], 6)
            self.assertEqual(validation["validated_outcomes"], 0)
            self.assertEqual(validation["matched_edges"], [])
            self.assertEqual(validation["status"], "INSUFFICIENT_PRIOR_CONDITION_HISTORY")


def _seed_history_db(path: Path) -> None:
    conn = sqlite3.connect(path)
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
        CREATE TABLE jsonl_events (
            source_name TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            event_type TEXT,
            timestamp_utc TEXT,
            pair TEXT,
            direction TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_name, line_no)
        );
        """
    )
    conn.executemany(
        "INSERT INTO legacy_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "trades",
                "1",
                "2026-04-01",
                "EUR_USD",
                "LONG",
                120.0,
                None,
                None,
                "continuation shelf",
                json.dumps(
                    {
                        "reason": "MARKET_ORDER",
                        "session_hour": 8,
                        "thesis_structure": "trend continuation",
                        "regime": "bear trend",
                    }
                ),
            ),
            (
                "trades",
                "2",
                "2026-04-01",
                "EUR_USD",
                "LONG",
                -80.0,
                None,
                None,
                "continuation shelf",
                json.dumps(
                    {
                        "reason": "MARKET_ORDER",
                        "session_hour": 9,
                        "thesis_structure": "trend continuation",
                        "regime": "TREND_UP",
                    }
                ),
            ),
            (
                "seat_outcomes",
                "3",
                "2026-04-01",
                "AUD_JPY",
                "SHORT",
                -50.0,
                "LIMIT",
                None,
                "range box",
                json.dumps({"setup_type": "Range-Mean-Revert", "session_bucket": "Tokyo", "regime": "quiet / stable"}),
            ),
        ],
    )
    conn.execute(
        "INSERT INTO jsonl_events VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "s_hunt_ledger",
            1,
            "snapshot",
            "2026-04-01T02:00:00+00:00",
            None,
            None,
            json.dumps(
                {
                    "state_last_updated": "2026-04-01T02:00:00+00:00",
                    "horizons": [
                        {
                            "pair": "AUD_JPY",
                            "direction": "SHORT",
                            "raw": "AUD_JPY SHORT / Range-Mean-Revert ceiling fade",
                            "orderability": "LIMIT",
                        },
                        {
                            "pair": "EUR_USD",
                            "direction": "LONG",
                            "raw": "EUR_USD LONG trend continuation",
                            "orderability": "MARKET",
                        },
                    ],
                }
            ),
        ),
    )
    conn.commit()
    conn.close()


def _seed_text_condition_history_db(path: Path) -> None:
    conn = sqlite3.connect(path)
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
        CREATE TABLE jsonl_events (
            source_name TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            event_type TEXT,
            timestamp_utc TEXT,
            pair TEXT,
            direction TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_name, line_no)
        );
        """
    )
    conn.execute(
        "INSERT INTO legacy_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "seat_outcomes",
            "1",
            "2026-04-18",
            "AUD_JPY",
            "SHORT",
            210.0,
            None,
            None,
            "paid upper rail is already armed",
            json.dumps(
                {
                    "created_at": "2026-04-18 14:05:00",
                    "orderability": "LIMIT",
                    "mtf_chain": "H1 corrective | M5 seat upper range | M1 trigger at the lid",
                    "why": "paid upper rail is already armed",
                    "trigger": "upper box rotation",
                }
            ),
        ),
    )
    conn.commit()
    conn.close()


def _seed_walk_forward_history_db(path: Path) -> None:
    conn = sqlite3.connect(path)
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
        CREATE TABLE jsonl_events (
            source_name TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            event_type TEXT,
            timestamp_utc TEXT,
            pair TEXT,
            direction TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_name, line_no)
        );
        """
    )
    rows = []
    for index, value in enumerate((100.0, 80.0, 70.0, 60.0, 50.0, 40.0), start=1):
        rows.append(
            (
                "trades",
                str(index),
                "2026-04-01",
                "EUR_USD",
                "LONG",
                value,
                None,
                None,
                "trend continuation",
                json.dumps(
                    {
                        "created_at": f"2026-04-01 0{index}:00:00",
                        "reason": "MARKET_ORDER",
                        "thesis_structure": "trend continuation",
                        "regime": "TREND_UP",
                    }
                ),
            )
        )
    for offset, value in enumerate((-100.0, -80.0, -70.0, -60.0, -50.0, -40.0), start=1):
        rows.append(
            (
                "trades",
                str(100 + offset),
                "2026-04-02",
                "AUD_JPY",
                "SHORT",
                value,
                None,
                None,
                "failed breakout rejection",
                json.dumps(
                    {
                        "created_at": f"2026-04-02 0{offset}:00:00",
                        "reason": "MARKET_ORDER",
                        "thesis_structure": "failed breakout rejection",
                        "regime": "RANGE",
                    }
                ),
            )
        )
    conn.executemany("INSERT INTO legacy_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def _seed_same_timestamp_history_db(path: Path) -> None:
    conn = sqlite3.connect(path)
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
        CREATE TABLE jsonl_events (
            source_name TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            event_type TEXT,
            timestamp_utc TEXT,
            pair TEXT,
            direction TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_name, line_no)
        );
        """
    )
    rows = [
        (
            "trades",
            str(index),
            "2026-04-01",
            "EUR_USD",
            "LONG",
            100.0,
            None,
            None,
            "trend continuation",
            json.dumps(
                {
                    "created_at": "2026-04-01 08:00:00",
                    "reason": "MARKET_ORDER",
                    "thesis_structure": "trend continuation",
                    "regime": "TREND_UP",
                }
            ),
        )
        for index in range(1, 7)
    ]
    conn.executemany("INSERT INTO legacy_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    unittest.main()
