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
            self.assertIn("context_feature_outcomes", payload)
            self.assertIn("context_feature_coverage_pct", payload)
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

    def test_execution_ledger_uses_original_position_side_not_close_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_empty_history_db(db)
            _seed_execution_ledger_with_close_side_rows(ledger)

            summary = OutcomeMartBuilder(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "outcome_mart.json",
                report_path=root / "outcome_mart.md",
            ).run()

            payload = json.loads((root / "outcome_mart.json").read_text())
            edges = {item["key"]: item for item in payload["pair_direction_edges"]}
            self.assertEqual(summary.execution_ledger_outcomes, 2)
            self.assertEqual(edges["GBP_USD:LONG:ALL:ALL:ALL:ALL"]["net_jpy"], -1200.0)
            self.assertEqual(edges["GBP_USD:LONG:ALL:ALL:ALL:ALL"]["execution_ledger_outcome_n"], 1)
            self.assertEqual(edges["AUD_JPY:SHORT:ALL:ALL:ALL:ALL"]["net_jpy"], 500.0)
            self.assertEqual(edges["AUD_JPY:SHORT:ALL:ALL:ALL:ALL"]["execution_ledger_outcome_n"], 1)
            self.assertNotIn("GBP_USD:SHORT:ALL:ALL:ALL:ALL", edges)
            self.assertNotIn("AUD_JPY:LONG:ALL:ALL:ALL:ALL", edges)

    def test_execution_ledger_context_features_are_joined_to_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_empty_history_db(db)
            _seed_execution_ledger_with_context_features(ledger)

            summary = OutcomeMartBuilder(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "outcome_mart.json",
                report_path=root / "outcome_mart.md",
            ).run()

            payload = json.loads((root / "outcome_mart.json").read_text())
            features = {item["key"]: item for item in payload["context_feature_edges"]}

            self.assertEqual(summary.execution_ledger_outcomes, 1)
            self.assertEqual(payload["source_coverage"]["context_feature_outcomes"], 1)
            self.assertEqual(payload["source_coverage"]["context_feature_coverage_pct"], 100.0)
            self.assertEqual(features["matrix_ref:matrix:EUR_USD:LONG"]["net_jpy"], 700.0)
            self.assertEqual(features["context_asset_ref:context_asset:XAU_USD"]["outcome_n"], 1)
            self.assertEqual(features["matrix_support_layer:context_asset_chart"]["net_jpy"], 700.0)
            self.assertEqual(features["news_context:news_theme_followthrough"]["net_jpy"], 700.0)
            report = (root / "outcome_mart.md").read_text()
            self.assertIn("Context Feature Edges", report)

    def test_execution_ledger_uses_broker_order_lane_when_gateway_receipt_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_empty_history_db(db)
            _seed_execution_ledger_with_broker_order_lane(ledger)

            summary = OutcomeMartBuilder(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "outcome_mart.json",
                report_path=root / "outcome_mart.md",
            ).run()

            payload = json.loads((root / "outcome_mart.json").read_text())
            edges = {item["key"]: item for item in payload["pair_direction_edges"]}
            method_edges = {item["key"]: item for item in payload["method_edges"]}
            self.assertEqual(summary.execution_ledger_outcomes, 1)
            self.assertEqual(edges["EUR_USD:LONG:ALL:ALL:ALL:ALL"]["execution_ledger_outcome_n"], 1)
            method_key = "EUR_USD:LONG:TREND_CONTINUATION:ALL:ALL:ALL"
            self.assertEqual(method_edges[method_key]["execution_ledger_outcome_n"], 1)
            self.assertEqual(method_edges[method_key]["net_jpy"], 300.0)


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


def _seed_empty_history_db(path: Path) -> None:
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
    conn.commit()
    conn.close()


def _seed_execution_ledger_with_close_side_rows(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE execution_events (
            ts_utc TEXT NOT NULL,
            event_type TEXT NOT NULL,
            lane_id TEXT,
            order_id TEXT,
            trade_id TEXT,
            pair TEXT,
            side TEXT,
            units INTEGER,
            realized_pl_jpy REAL,
            raw_json TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        """
        INSERT INTO execution_events (
            ts_utc, event_type, lane_id, order_id, trade_id, pair, side, units,
            realized_pl_jpy, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "2026-05-13T00:00:00Z",
                "ORDER_FILLED",
                "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
                None,
                "long-loss",
                "GBP_USD",
                "LONG",
                8000,
                None,
                json.dumps({"reason": "MARKET_ORDER"}),
            ),
            (
                "2026-05-13T01:00:00Z",
                "TRADE_CLOSED",
                "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
                None,
                "long-loss",
                "GBP_USD",
                "SHORT",
                8000,
                -1200.0,
                json.dumps({"reason": "MARKET_ORDER_TRADE_CLOSE"}),
            ),
            (
                "2026-05-13T02:00:00Z",
                "ORDER_FILLED",
                "range_trader:AUD_JPY:SHORT:RANGE_ROTATION",
                None,
                "short-win",
                "AUD_JPY",
                "SHORT",
                -3000,
                None,
                json.dumps({"reason": "LIMIT_ORDER"}),
            ),
            (
                "2026-05-13T03:00:00Z",
                "TRADE_REDUCED",
                "range_trader:AUD_JPY:SHORT:RANGE_ROTATION",
                None,
                "short-win",
                "AUD_JPY",
                "LONG",
                3000,
                500.0,
                json.dumps({"reason": "TRADE_CLOSE"}),
            ),
        ],
    )
    conn.commit()
    conn.close()


def _seed_execution_ledger_with_context_features(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE execution_events (
            ts_utc TEXT NOT NULL,
            event_type TEXT NOT NULL,
            lane_id TEXT,
            order_id TEXT,
            trade_id TEXT,
            pair TEXT,
            side TEXT,
            units INTEGER,
            realized_pl_jpy REAL,
            raw_json TEXT NOT NULL
        );
        """
    )
    context_evidence = {
        "market_context_matrix_ref": "matrix:EUR_USD:LONG",
        "matrix_support_layers": ["context_asset_chart"],
        "context_asset_refs": ["context_asset:XAU_USD"],
        "context_asset_symbols": ["XAU_USD"],
        "news_context": ["forecast_drivers_for=[\"news_theme_followthrough USD soft\"]"],
    }
    conn.executemany(
        """
        INSERT INTO execution_events (
            ts_utc, event_type, lane_id, order_id, trade_id, pair, side, units,
            realized_pl_jpy, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "2026-05-14T00:00:00Z",
                "GATEWAY_ORDER_SENT",
                "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                None,
                "ctx-win",
                "EUR_USD",
                "LONG",
                1000,
                None,
                json.dumps(
                    {
                        "entry_thesis_record": {
                            "status": "RECORDED",
                            "thesis": {
                                "trade_id": "ctx-win",
                                "context_evidence": context_evidence,
                            },
                        }
                    }
                ),
            ),
            (
                "2026-05-14T00:00:00Z",
                "ORDER_FILLED",
                "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                None,
                "ctx-win",
                "EUR_USD",
                "LONG",
                1000,
                None,
                json.dumps({"reason": "MARKET_ORDER"}),
            ),
            (
                "2026-05-14T01:00:00Z",
                "TRADE_CLOSED",
                "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                None,
                "ctx-win",
                "EUR_USD",
                "SHORT",
                1000,
                700.0,
                json.dumps({"reason": "MARKET_ORDER_TRADE_CLOSE"}),
            ),
        ],
    )
    conn.commit()
    conn.close()


def _seed_execution_ledger_with_broker_order_lane(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE execution_events (
            ts_utc TEXT NOT NULL,
            event_type TEXT NOT NULL,
            lane_id TEXT,
            order_id TEXT,
            trade_id TEXT,
            pair TEXT,
            side TEXT,
            units INTEGER,
            realized_pl_jpy REAL,
            raw_json TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        """
        INSERT INTO execution_events (
            ts_utc, event_type, lane_id, order_id, trade_id, pair, side, units,
            realized_pl_jpy, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "2026-05-15T00:00:00Z",
                "ORDER_ACCEPTED",
                "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                "order-broker",
                None,
                "EUR_USD",
                "LONG",
                1000,
                None,
                json.dumps({"clientExtensions": {"comment": "qr-vnext lane=trend_trader:EUR_USD:LONG:TREND_CONTINUATION"}}),
            ),
            (
                "2026-05-15T00:00:01Z",
                "ORDER_FILLED",
                None,
                "order-broker",
                "broker-trade",
                "EUR_USD",
                "LONG",
                1000,
                None,
                json.dumps({"reason": "MARKET_ORDER"}),
            ),
            (
                "2026-05-15T00:20:00Z",
                "TRADE_CLOSED",
                None,
                "close-broker",
                "broker-trade",
                "EUR_USD",
                "SHORT",
                1000,
                300.0,
                json.dumps({"reason": "TAKE_PROFIT_ORDER"}),
            ),
        ],
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
