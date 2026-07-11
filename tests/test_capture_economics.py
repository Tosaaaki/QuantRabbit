"""Unit tests for capture_economics.py."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.capture_economics import (
    build_capture_economics,
    read_attributed_net_outcomes,
    read_attributed_system_entries,
)


def _entry_fill_raw(
    *,
    timestamp: str,
    pair: str,
    side: str,
    order_id: str,
    trade_id: str,
    reason: str,
    units: int = 1000,
) -> dict:
    signed_units = abs(units) if side == "LONG" else -abs(units)
    return {
        "id": f"fill-{trade_id}",
        "time": timestamp,
        "type": "ORDER_FILL",
        "orderID": order_id,
        "instrument": pair,
        "units": str(signed_units),
        "reason": reason,
        "tradeOpened": {
            "tradeID": trade_id,
            "units": str(signed_units),
        },
    }


def _make_db(path: Path, closes: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE execution_events (
            event_uid TEXT PRIMARY KEY,
            ts_utc TEXT, event_type TEXT, lane_id TEXT, order_id TEXT,
            trade_id TEXT, pair TEXT, side TEXT, units INTEGER,
            realized_pl_jpy REAL, financing_jpy REAL, exit_reason TEXT,
            raw_json TEXT
        )
        """
    )
    conn.execute(
        "CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT, updated_at_utc TEXT)"
    )
    conn.execute(
        "INSERT INTO sync_state VALUES ('oanda_transaction_coverage_start_utc', ?, ?)",
        ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00"),
    )
    rows = []
    for i, c in enumerate(closes):
        trade_id = c.get("trade_id", f"t{i}")
        side = c.get("side", "LONG")
        entry_units = int(c.get("entry_units", 1000))
        signed_entry_units = abs(entry_units) if side == "LONG" else -abs(entry_units)
        entry_ts_utc = c.get("entry_ts_utc", c["ts_utc"])
        method = c.get("method", "RANGE_ROTATION")
        lane_id = c.get("lane_id", f"range_trader:{c['pair']}:{side}:{method}")
        entry_reason = c.get("entry_reason", "LIMIT_ORDER")
        if c.get("attributed", True):
            rows.append(
                (f"g{i}", entry_ts_utc, "GATEWAY_ORDER_SENT", lane_id, f"o{i}", trade_id,
                 c["pair"], side, signed_entry_units, None, 0.0, entry_reason,
                 json.dumps({"type": entry_reason}))
            )
            rows.append(
                (f"f{i}", entry_ts_utc, "ORDER_FILLED", None, f"o{i}", trade_id,
                 c["pair"], side, signed_entry_units, None, 0.0, entry_reason,
                 json.dumps(
                     _entry_fill_raw(
                         timestamp=c.get("broker_entry_ts_utc", entry_ts_utc),
                         pair=c["pair"],
                         side=side,
                         order_id=f"o{i}",
                         trade_id=trade_id,
                         reason=entry_reason,
                         units=entry_units,
                     )
                 ))
            )
        exit_reason = c.get("exit_reason", "TAKE_PROFIT_ORDER")
        financing = c.get("financing", 0.0)
        close_raw = c.get(
            "raw_json",
            {
                "time": c.get("broker_close_ts_utc", c["ts_utc"]),
                "type": "ORDER_FILL",
                "instrument": c["pair"],
                "orderID": f"x{i}",
                "reason": exit_reason,
                "commission": "0.0",
                "guaranteedExecutionFee": "0.0",
                (
                    "tradesClosed"
                    if c.get("event_type", "TRADE_CLOSED") == "TRADE_CLOSED"
                    else "tradeReduced"
                ): (
                    [
                        {
                            "tradeID": trade_id,
                            "realizedPL": str(c["pl"]),
                            "financing": str(financing),
                        }
                    ]
                    if c.get("event_type", "TRADE_CLOSED") == "TRADE_CLOSED"
                    else {
                        "tradeID": trade_id,
                        "realizedPL": str(c["pl"]),
                        "financing": str(financing),
                    }
                ),
            },
        )
        rows.append(
            (f"c{i}", c["ts_utc"], c.get("event_type", "TRADE_CLOSED"), None, f"x{i}", trade_id,
             c["pair"], "SHORT", 1000, c["pl"], financing,
             exit_reason, json.dumps(close_raw))
        )
    conn.executemany(
        "INSERT INTO execution_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()


class CaptureEconomicsTest(unittest.TestCase):
    def test_oanda_nanosecond_timestamps_are_audited_on_python_39(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": "2026-05-06T15:30:50.236779206Z",
                        "pair": "EUR_USD",
                        "pl": 125.0,
                        "trade_id": "470226",
                    }
                ],
            )

            outcomes = read_attributed_net_outcomes(db)
            entries = read_attributed_system_entries(db)

            self.assertIsNotNone(outcomes)
            self.assertEqual([row.trade_id for row in outcomes or []], ["470226"])
            self.assertIsNotNone(entries)
            entry = (entries or [])[0]
            outcome = (outcomes or [])[0]
            self.assertEqual(
                entry.broker_entry_ts_utc,
                "2026-05-06T15:30:50.236779206Z",
            )
            self.assertTrue(entry.broker_time_consistent)
            self.assertEqual(
                outcome.broker_close_ts_utc,
                "2026-05-06T15:30:50.236779206Z",
            )
            self.assertTrue(outcome.broker_time_consistent)

    def test_entry_fill_raw_truth_must_match_every_normalized_identity_field(self) -> None:
        cases = {
            "time": lambda raw: raw.update(time="2026-06-02T10:00:01Z"),
            "instrument": lambda raw: raw.update(instrument="GBP_USD"),
            "order_id": lambda raw: raw.update(orderID="other-order"),
            "trade_id": lambda raw: raw["tradeOpened"].update(tradeID="other-trade"),
            "signed_units": lambda raw: raw["tradeOpened"].update(units="-1000"),
            "order_units_sign": lambda raw: raw.update(units="-1000"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, mutate in cases.items():
                with self.subTest(field=name):
                    db = root / f"{name}.db"
                    _make_db(
                        db,
                        [
                            {
                                "entry_ts_utc": "2026-06-02T10:00:00Z",
                                "ts_utc": "2026-06-02T10:30:00Z",
                                "pair": "EUR_USD",
                                "pl": 100.0,
                            }
                        ],
                    )
                    with sqlite3.connect(db) as conn:
                        raw = json.loads(
                            conn.execute(
                                "SELECT raw_json FROM execution_events WHERE event_type='ORDER_FILLED'"
                            ).fetchone()[0]
                        )
                        mutate(raw)
                        conn.execute(
                            "UPDATE execution_events SET raw_json=? WHERE event_type='ORDER_FILLED'",
                            (json.dumps(raw),),
                        )
                    self.assertIsNone(read_attributed_system_entries(db))
                    self.assertIsNone(read_attributed_net_outcomes(db))

            for name, assignment in {
                "normalized_side": "side='SHORT'",
                "normalized_units": "units=999",
            }.items():
                with self.subTest(field=name):
                    db = root / f"{name}.db"
                    _make_db(
                        db,
                        [
                            {
                                "entry_ts_utc": "2026-06-02T10:00:00Z",
                                "ts_utc": "2026-06-02T10:30:00Z",
                                "pair": "EUR_USD",
                                "pl": 100.0,
                            }
                        ],
                    )
                    with sqlite3.connect(db) as conn:
                        conn.execute(
                            f"UPDATE execution_events SET {assignment} WHERE event_type='ORDER_FILLED'"
                        )
                    self.assertIsNone(read_attributed_system_entries(db))

    def test_system_gateway_candidate_with_ambiguous_fill_fails_whole_entry_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "entry_ts_utc": "2026-06-02T10:00:00Z",
                        "ts_utc": "2026-06-02T10:30:00Z",
                        "pair": "EUR_USD",
                        "pl": 100.0,
                    }
                ],
            )
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE execution_events SET trade_id=NULL WHERE event_type='ORDER_FILLED'"
                )

            self.assertIsNone(read_attributed_system_entries(db))

    def test_raw_order_id_keeps_broken_gateway_fill_from_being_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "entry_ts_utc": "2026-06-02T10:00:00Z",
                        "ts_utc": "2026-06-02T10:30:00Z",
                        "pair": "EUR_USD",
                        "pl": -1000.0,
                    }
                ],
            )
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    UPDATE execution_events
                    SET lane_id=NULL, order_id=NULL
                    WHERE event_type='ORDER_FILLED'
                    """
                )

            self.assertIsNone(read_attributed_system_entries(db))
            self.assertIsNone(read_attributed_net_outcomes(db))

    def test_manual_raw_tag_cannot_hide_a_gateway_attributed_system_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "entry_ts_utc": "2026-06-02T10:00:00Z",
                        "ts_utc": "2026-06-02T10:30:00Z",
                        "pair": "EUR_USD",
                        "pl": -1000.0,
                    }
                ],
            )
            with sqlite3.connect(db) as conn:
                raw = json.loads(
                    conn.execute(
                        """
                        SELECT raw_json FROM execution_events
                        WHERE event_type='ORDER_FILLED'
                        """
                    ).fetchone()[0]
                )
                raw["tradeOpened"]["tradeClientExtensions"] = {"tag": "manual"}
                conn.execute(
                    """
                    UPDATE execution_events
                    SET lane_id=NULL, raw_json=?
                    WHERE event_type='ORDER_FILLED'
                    """,
                    (json.dumps(raw),),
                )

            self.assertIsNone(read_attributed_system_entries(db))
            self.assertIsNone(read_attributed_net_outcomes(db))

    def test_final_close_raw_broker_time_must_match_normalized_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "entry_ts_utc": "2026-06-02T10:00:00Z",
                        "ts_utc": "2026-06-02T10:30:00Z",
                        "pair": "EUR_USD",
                        "pl": 100.0,
                    }
                ],
            )
            with sqlite3.connect(db) as conn:
                raw = json.loads(
                    conn.execute(
                        "SELECT raw_json FROM execution_events WHERE event_type='TRADE_CLOSED'"
                    ).fetchone()[0]
                )
                raw["time"] = "2026-06-02T10:30:00.000000001Z"
                conn.execute(
                    "UPDATE execution_events SET raw_json=? WHERE event_type='TRADE_CLOSED'",
                    (json.dumps(raw),),
                )

            self.assertIsNone(read_attributed_net_outcomes(db))

    def test_gateway_vehicle_lane_and_fill_parent_lane_are_same_entry_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": "2026-05-06T15:31:00.123456789Z",
                        "pair": "EUR_JPY",
                        "side": "LONG",
                        "method": "TREND_CONTINUATION",
                        "lane_id": (
                            "trend_trader:EUR_JPY:LONG:TREND_CONTINUATION:MARKET"
                        ),
                        "entry_reason": "MARKET_ORDER",
                        "pl": 250.0,
                        "trade_id": "470243",
                    }
                ],
            )
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    UPDATE execution_events
                    SET lane_id='trend_trader:EUR_JPY:LONG:TREND_CONTINUATION'
                    WHERE event_uid='f0'
                    """
                )

            outcomes = read_attributed_net_outcomes(db)

            self.assertIsNotNone(outcomes)
            self.assertEqual(len(outcomes or []), 1)
            outcome = (outcomes or [])[0]
            self.assertEqual(outcome.entry_vehicle, "MARKET")
            self.assertTrue(outcome.entry_truth_consistent)

    def test_raw_multiclose_components_and_normalized_events_must_match_one_to_one(self) -> None:
        """A missing sibling loss must not turn one broker close into a win.

        This models an ORDER_FILL that closed two system trades while a
        parser/legacy defect persisted only the winning normalized row.  The
        complete raw transaction is available on that surviving row, so the
        canonical reader must reject missing, duplicate, extra, and
        amount-divergent normalized events rather than selectively counting it.
        """

        cases = {
            "valid": (("t1", 100.0), ("t2", -1000.0)),
            "missing_sibling_loss": (("t1", 100.0),),
            "duplicate_winner": (
                ("t1", 100.0),
                ("t1", 100.0),
                ("t2", -1000.0),
            ),
            "extra_event": (
                ("t1", 100.0),
                ("t2", -1000.0),
                ("t3", 50.0),
            ),
            "amount_mismatch": (("t1", 101.0), ("t2", -1000.0)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case_name, actual_closes in cases.items():
                with self.subTest(case=case_name):
                    db = root / f"{case_name}.db"
                    conn = sqlite3.connect(db)
                    conn.execute(
                        """
                        CREATE TABLE execution_events (
                            event_uid TEXT PRIMARY KEY,
                            ts_utc TEXT,
                            event_type TEXT,
                            lane_id TEXT,
                            order_id TEXT,
                            trade_id TEXT,
                            pair TEXT,
                            side TEXT,
                            units INTEGER,
                            realized_pl_jpy REAL,
                            financing_jpy REAL,
                            exit_reason TEXT,
                            raw_json TEXT
                        )
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE sync_state (
                            key TEXT PRIMARY KEY,
                            value TEXT,
                            updated_at_utc TEXT
                        )
                        """
                    )
                    conn.execute(
                        """
                        INSERT INTO sync_state VALUES (
                            'oanda_transaction_coverage_start_utc',
                            '2026-01-01T00:00:00Z',
                            '2026-01-01T00:00:00Z'
                        )
                        """
                    )
                    lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
                    for index, trade_id in enumerate(("t1", "t2", "t3"), start=1):
                        conn.execute(
                            """
                            INSERT INTO execution_events VALUES (
                                ?, '2026-07-01T00:00:00Z', 'GATEWAY_ORDER_SENT',
                                ?, ?, ?, 'EUR_USD', 'LONG', 1000,
                                NULL, 0.0, 'LIMIT_ORDER', ?
                            )
                            """,
                            (
                                f"gateway-{index}",
                                lane,
                                f"entry-order-{index}",
                                trade_id,
                                json.dumps({"type": "LIMIT_ORDER"}),
                            ),
                        )
                        conn.execute(
                            """
                            INSERT INTO execution_events VALUES (
                                ?, '2026-07-01T00:00:01Z', 'ORDER_FILLED',
                                ?, ?, ?, 'EUR_USD', 'LONG', 1000,
                                NULL, 0.0, 'LIMIT_ORDER', ?
                            )
                            """,
                            (
                                f"fill-{index}",
                                lane,
                                f"entry-order-{index}",
                                trade_id,
                                json.dumps(
                                    _entry_fill_raw(
                                        timestamp="2026-07-01T00:00:01Z",
                                        pair="EUR_USD",
                                        side="LONG",
                                        order_id=f"entry-order-{index}",
                                        trade_id=trade_id,
                                        reason="LIMIT_ORDER",
                                    )
                                ),
                            ),
                        )

                    close_raw = {
                        "id": "close-transaction-1",
                        "time": "2026-07-01T01:00:00Z",
                        "type": "ORDER_FILL",
                        "instrument": "EUR_USD",
                        "orderID": "close-transaction-1",
                        "reason": "MARKET_ORDER_TRADE_CLOSE",
                        "commission": "0.0",
                        "guaranteedExecutionFee": "0.0",
                        "tradesClosed": [
                            {
                                "tradeID": "t1",
                                "realizedPL": "100.0",
                                "financing": "0.0",
                            },
                            {
                                "tradeID": "t2",
                                "realizedPL": "-1000.0",
                                "financing": "0.0",
                            },
                        ],
                    }
                    for index, (trade_id, realized) in enumerate(actual_closes):
                        conn.execute(
                            """
                            INSERT INTO execution_events VALUES (
                                ?, '2026-07-01T01:00:00Z', 'TRADE_CLOSED',
                                NULL, 'close-transaction-1', ?, 'EUR_USD', 'SHORT',
                                1000, ?, 0.0, 'MARKET_ORDER_TRADE_CLOSE', ?
                            )
                            """,
                            (
                                f"close-{index}",
                                trade_id,
                                realized,
                                json.dumps(close_raw),
                            ),
                        )
                    conn.commit()
                    conn.close()

                    outcomes = read_attributed_net_outcomes(db)
                    if case_name == "valid":
                        self.assertIsNotNone(outcomes)
                        self.assertEqual(
                            [(row.trade_id, row.realized_pl_jpy) for row in outcomes or []],
                            [("t1", 100.0), ("t2", -1000.0)],
                        )
                    else:
                        self.assertIsNone(outcomes)

    def test_authoritative_oanda_close_transaction_cannot_disappear_entirely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(db, [])
            raw = {
                "id": "broker-close-1",
                "type": "ORDER_FILL",
                "reason": "MARKET_ORDER_TRADE_CLOSE",
                "tradesClosed": [
                    {
                        "tradeID": "missing-event-trade",
                        "realizedPL": "-500.0",
                        "financing": "0.0",
                    }
                ],
            }
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    CREATE TABLE oanda_transactions (
                        transaction_id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        raw_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO oanda_transactions VALUES (?, 'ORDER_FILL', ?)",
                    ("broker-close-1", json.dumps(raw)),
                )

            self.assertIsNone(read_attributed_net_outcomes(db))

    def test_missing_ledger_reports_low_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_capture_economics(
                ledger_path=root / "missing.db",
                output_path=root / "out.json",
                report_path=root / "report.md",
            )
            self.assertEqual(summary.status, "LOW_SAMPLE")
            self.assertEqual(summary.trades, 0)

    def test_negative_expectancy_flagged_from_payoff_vs_breakeven(self) -> None:
        """Model of the 2026-05/06 ledger: high win rate, tiny wins, big losses."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closes = []
            for i in range(21):
                closes.append({"ts_utc": f"2026-06-0{(i % 5) + 1}T10:{i:02d}:00Z", "pair": "EUR_USD", "pl": 300.0})
            for i in range(9):
                closes.append(
                    {"ts_utc": f"2026-06-0{(i % 5) + 1}T12:{i:02d}:00Z", "pair": "EUR_USD",
                     "pl": -1500.0, "exit_reason": "MARKET_ORDER_TRADE_CLOSE"}
                )
            db = root / "ledger.db"
            _make_db(db, closes)
            summary = build_capture_economics(
                ledger_path=db, output_path=root / "out.json", report_path=root / "report.md"
            )
            self.assertEqual(summary.status, "NEGATIVE_EXPECTANCY")
            self.assertEqual(summary.trades, 30)
            self.assertAlmostEqual(summary.win_rate or 0, 0.7)
            self.assertAlmostEqual(summary.payoff_ratio or 0, 0.2)
            # breakeven at 70% win rate = 0.3/0.7 ≈ 0.4286 > payoff 0.2
            self.assertAlmostEqual(summary.breakeven_payoff or 0, 0.4286, places=3)
            payload = json.loads((root / "out.json").read_text())
            self.assertIn("MARKET_ORDER_TRADE_CLOSE", payload["by_exit_reason"])
            self.assertEqual(
                payload["repair_summary"]["dominant_loss_exit_reason"],
                "MARKET_ORDER_TRADE_CLOSE",
            )
            self.assertEqual(
                payload["by_pair_side_exit_reason"]["EUR_USD"]["LONG"]["MARKET_ORDER_TRADE_CLOSE"]["losses"],
                9,
            )
            self.assertEqual(
                payload["by_pair_side_method_exit_reason"]["EUR_USD"]["LONG"]["RANGE_ROTATION"][
                    "TAKE_PROFIT_ORDER"
                ]["wins"],
                21,
            )
            self.assertGreater(payload["repair_summary"]["payoff_gap_to_breakeven"], 0)
            self.assertTrue(
                any("MARKET_ORDER_TRADE_CLOSE drag" in item for item in payload["action_items"])
            )
            report = (root / "report.md").read_text()
            self.assertIn("## Repair Summary", report)
            self.assertIn("MARKET_ORDER_TRADE_CLOSE", report)

    def test_partial_closes_aggregate_to_one_trade_outcome(self) -> None:
        """TRADE_REDUCED milestones + final TRADE_CLOSED on the same trade_id
        are ONE trade whose realized P/L is the sum — two +300 partials and a
        -900 final close is a single -300 LOSS, not two wins and a loss
        (2026-06-10 audit finding: per-event counting inflated win rate)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closes = [
                {"ts_utc": "2026-06-02T10:00:00Z", "pair": "EUR_USD", "pl": 300.0,
                 "trade_id": "t1", "event_type": "TRADE_REDUCED"},
                {"ts_utc": "2026-06-02T11:00:00Z", "pair": "EUR_USD", "pl": 300.0,
                 "trade_id": "t1", "event_type": "TRADE_REDUCED"},
                {"ts_utc": "2026-06-02T12:00:00Z", "pair": "EUR_USD", "pl": -900.0,
                 "trade_id": "t1", "exit_reason": "MARKET_ORDER_TRADE_CLOSE"},
                # An unrelated clean win on its own trade.
                {"ts_utc": "2026-06-02T13:00:00Z", "pair": "GBP_USD", "pl": 500.0, "trade_id": "t2"},
            ]
            db = root / "ledger.db"
            _make_db(db, closes)
            summary = build_capture_economics(
                ledger_path=db, output_path=root / "out.json", report_path=root / "report.md"
            )
            self.assertEqual(summary.trades, 2)
            payload = json.loads((root / "out.json").read_text())
            self.assertEqual(payload["overall"]["wins"], 1)
            self.assertEqual(payload["overall"]["losses"], 1)
            self.assertAlmostEqual(payload["overall"]["net_jpy"], 200.0)
            # The aggregated trade's exit reason is its FINAL close event.
            self.assertIn("MARKET_ORDER_TRADE_CLOSE", payload["by_exit_reason"])
            self.assertEqual(payload["by_exit_reason"]["MARKET_ORDER_TRADE_CLOSE"]["losses"], 1)
            self.assertEqual(
                payload["by_pair_side_method_exit_reason"]["EUR_USD"]["LONG"]["RANGE_ROTATION"][
                    "MARKET_ORDER_TRADE_CLOSE"
                ]["losses"],
                1,
            )

    def test_unresolved_partial_only_trade_is_not_an_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": "2026-06-02T10:00:00Z",
                        "pair": "EUR_USD",
                        "pl": 300.0,
                        "trade_id": "still-open",
                        "event_type": "TRADE_REDUCED",
                    }
                ],
            )

            summary = build_capture_economics(
                ledger_path=db,
                output_path=root / "out.json",
                report_path=root / "report.md",
            )

            self.assertEqual(summary.trades, 0)

    def test_close_and_daily_financing_are_included_in_net_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": "2026-06-02T10:00:00Z",
                        "pair": "EUR_USD",
                        "pl": 100.0,
                        "financing": -25.0,
                        "trade_id": "financed-trade",
                    }
                ],
            )
            financing = {
                "type": "DAILY_FINANCING",
                "financing": "-100.0",
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {"tradeID": "financed-trade", "financing": "-100.0"}
                        ]
                    }
                ],
            }
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id,
                        trade_id, pair, side, units, realized_pl_jpy,
                        financing_jpy, exit_reason, raw_json
                    ) VALUES (?, ?, 'OANDA_TRANSACTION', NULL, NULL,
                              NULL, NULL, NULL, NULL, NULL, ?, NULL, ?)
                    """,
                    ("daily-financing", "2026-06-02T09:00:00Z", -100.0, json.dumps(financing)),
                )

            summary = build_capture_economics(
                ledger_path=db,
                output_path=root / "out.json",
                report_path=root / "report.md",
            )

            payload = json.loads((root / "out.json").read_text())
            self.assertEqual(summary.trades, 1)
            self.assertEqual(payload["overall"]["wins"], 0)
            self.assertEqual(payload["overall"]["losses"], 1)
            self.assertEqual(payload["overall"]["net_jpy"], -25.0)

    def test_financing_audit_failure_is_unreadable_not_empty_low_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(
                db,
                [{"ts_utc": "2026-06-02T10:00:00Z", "pair": "EUR_USD", "pl": 100.0}],
            )
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id,
                        trade_id, pair, side, units, realized_pl_jpy,
                        financing_jpy, exit_reason, raw_json
                    ) VALUES (?, ?, 'OANDA_TRANSACTION', NULL, NULL,
                              NULL, NULL, NULL, NULL, NULL, ?, NULL, ?)
                    """,
                    (
                        "unsupported-cash",
                        "2026-06-02T09:00:00Z",
                        -500.0,
                        json.dumps({"type": "DIVIDEND_ADJUSTMENT"}),
                    ),
                )

            self.assertIsNone(read_attributed_net_outcomes(db))
            summary = build_capture_economics(
                ledger_path=db,
                output_path=root / "out.json",
                report_path=root / "report.md",
            )
            self.assertEqual(summary.status, "EVIDENCE_UNREADABLE")
            self.assertEqual(json.loads((root / "out.json").read_text())["evidence_read_status"], "FAILED")

    def test_zero_account_financing_still_allocates_offsetting_trade_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": "2026-06-02T10:00:00Z",
                        "pair": "EUR_USD",
                        "pl": 100.0,
                        "trade_id": "system-trade",
                    }
                ],
            )
            financing = {
                "type": "DAILY_FINANCING",
                "financing": "0.0",
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {"tradeID": "system-trade", "financing": "-200.0"},
                            {"tradeID": "manual-trade", "financing": "200.0"},
                        ]
                    }
                ],
            }
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id,
                        trade_id, pair, side, units, realized_pl_jpy,
                        financing_jpy, exit_reason, raw_json
                    ) VALUES (?, ?, 'OANDA_TRANSACTION', NULL, NULL,
                              NULL, NULL, NULL, NULL, NULL, 0.0, NULL, ?)
                    """,
                    ("offset-financing", "2026-06-02T09:00:00Z", json.dumps(financing)),
                )

            rows = read_attributed_net_outcomes(db)

            self.assertIsNotNone(rows)
            assert rows is not None
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].realized_pl_jpy, -100.0)

    def test_valid_empty_ledger_is_empty_not_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            _make_db(db, [])

            self.assertEqual(read_attributed_net_outcomes(db), [])

    def test_operator_manual_lane_and_raw_owner_tag_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual_lane_db = root / "manual-lane.db"
            _make_db(
                manual_lane_db,
                [
                    {
                        "ts_utc": "2026-06-02T10:00:00Z",
                        "pair": "EUR_USD",
                        "pl": 500.0,
                        "lane_id": "operator_manual:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                    }
                ],
            )
            self.assertEqual(read_attributed_net_outcomes(manual_lane_db), [])

            raw_owner_db = root / "manual-tag.db"
            _make_db(
                raw_owner_db,
                [
                    {
                        "ts_utc": "2026-06-02T10:00:00Z",
                        "pair": "EUR_USD",
                        "pl": 500.0,
                    }
                ],
            )
            with sqlite3.connect(raw_owner_db) as conn:
                # A raw manual-owner tag is authoritative only when there is no
                # independent system gateway/lane receipt for the fill.
                conn.execute(
                    "DELETE FROM execution_events WHERE event_type='GATEWAY_ORDER_SENT'"
                )
                conn.execute(
                    "UPDATE execution_events SET raw_json=? WHERE event_type='ORDER_FILLED'",
                    (
                        json.dumps(
                            {
                                "type": "ORDER_FILL",
                                "reason": "LIMIT_ORDER",
                                "tradeOpened": {
                                    "tradeClientExtensions": {"tag": "operator_manual"}
                                },
                            }
                        ),
                    ),
                )
            self.assertEqual(read_attributed_net_outcomes(raw_owner_db), [])

    def test_later_or_trade_only_gateway_cannot_attribute_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": "2026-06-02T10:00:00Z",
                        "pair": "EUR_USD",
                        "pl": 500.0,
                    }
                ],
            )
            with sqlite3.connect(db) as conn:
                # Remove the valid pre-fill receipt, then append receipts that
                # historical trade-id fallback incorrectly treated as entry
                # attribution. Both rows occur after the fill.
                conn.execute("DELETE FROM execution_events WHERE event_uid='g0'")
                conn.execute(
                    """
                    INSERT INTO execution_events VALUES (
                        'late-same-order', '2026-06-02T10:01:00Z', 'ORDER_ACCEPTED',
                        'range_trader:EUR_USD:LONG:RANGE_ROTATION:LIMIT', 'o0', NULL,
                        'EUR_USD', 'LONG', 1000, NULL, 0.0, 'LIMIT_ORDER', '{}'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO execution_events VALUES (
                        'late-trade-only', '2026-06-02T10:02:00Z', 'ORDER_ACCEPTED',
                        'range_trader:EUR_USD:LONG:RANGE_ROTATION:LIMIT', 'other-order', 't0',
                        'EUR_USD', 'LONG', 1000, NULL, 0.0, 'LIMIT_ORDER', '{}'
                    )
                    """
                )

            self.assertEqual(read_attributed_net_outcomes(db), [])

    def test_invalid_or_mismatched_close_raw_makes_evidence_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid_db = root / "invalid.db"
            _make_db(
                invalid_db,
                [{"ts_utc": "2026-06-02T10:00:00Z", "pair": "EUR_USD", "pl": 100.0}],
            )
            with sqlite3.connect(invalid_db) as conn:
                conn.execute(
                    "UPDATE execution_events SET raw_json='{}' WHERE event_type='TRADE_CLOSED'"
                )
            self.assertIsNone(read_attributed_net_outcomes(invalid_db))

            mismatch_db = root / "mismatch.db"
            _make_db(
                mismatch_db,
                [{"ts_utc": "2026-06-02T10:00:00Z", "pair": "EUR_USD", "pl": 100.0}],
            )
            with sqlite3.connect(mismatch_db) as conn:
                raw = json.loads(
                    conn.execute(
                        "SELECT raw_json FROM execution_events WHERE event_type='TRADE_CLOSED'"
                    ).fetchone()[0]
                )
                raw["tradesClosed"][0]["realizedPL"] = "99.0"
                conn.execute(
                    "UPDATE execution_events SET raw_json=? WHERE event_type='TRADE_CLOSED'",
                    (json.dumps(raw),),
                )
            self.assertIsNone(read_attributed_net_outcomes(mismatch_db))

    def test_malformed_fill_raw_and_nonzero_close_commission_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed_db = root / "malformed-fill.db"
            _make_db(
                malformed_db,
                [{"ts_utc": "2026-06-02T10:00:00Z", "pair": "EUR_USD", "pl": 100.0}],
            )
            with sqlite3.connect(malformed_db) as conn:
                conn.execute(
                    "UPDATE execution_events SET raw_json='{bad' WHERE event_type='ORDER_FILLED'"
                )
            self.assertIsNone(read_attributed_net_outcomes(malformed_db))

            cash_db = root / "unsupported-close-cash.db"
            _make_db(
                cash_db,
                [{"ts_utc": "2026-06-02T10:00:00Z", "pair": "EUR_USD", "pl": 100.0}],
            )
            with sqlite3.connect(cash_db) as conn:
                raw = json.loads(
                    conn.execute(
                        "SELECT raw_json FROM execution_events WHERE event_type='TRADE_CLOSED'"
                    ).fetchone()[0]
                )
                raw["commission"] = "-5.0"
                conn.execute(
                    "UPDATE execution_events SET raw_json=? WHERE event_type='TRADE_CLOSED'",
                    (json.dumps(raw),),
                )
            self.assertIsNone(read_attributed_net_outcomes(cash_db))

    def test_contradictory_system_lane_and_fill_truth_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": "2026-06-02T10:00:00Z",
                        "pair": "GBP_USD",
                        "side": "LONG",
                        "pl": 500.0,
                        "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION:LIMIT",
                    }
                ],
            )

            self.assertIsNone(read_attributed_net_outcomes(db))

    def test_missing_or_post_entry_financing_coverage_marker_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_marker_db = root / "missing-marker.db"
            _make_db(
                missing_marker_db,
                [{"ts_utc": "2026-06-02T10:00:00Z", "pair": "EUR_USD", "pl": 100.0}],
            )
            with sqlite3.connect(missing_marker_db) as conn:
                conn.execute(
                    "DELETE FROM sync_state WHERE key='oanda_transaction_coverage_start_utc'"
                )
            self.assertIsNone(read_attributed_net_outcomes(missing_marker_db))

            precoverage_db = root / "precoverage.db"
            _make_db(
                precoverage_db,
                [{"ts_utc": "2026-06-02T10:00:00Z", "pair": "EUR_USD", "pl": 100.0}],
            )
            with sqlite3.connect(precoverage_db) as conn:
                conn.execute(
                    """
                    UPDATE sync_state SET value=?
                    WHERE key='oanda_transaction_coverage_start_utc'
                    """,
                    ("2026-06-02T10:00:01+00:00",),
                )
            self.assertIsNone(read_attributed_net_outcomes(precoverage_db))

    def test_segment_repair_priorities_split_tp_proof_from_market_close_leak(self) -> None:
        """High-rotation candidates must be scoped to the exact TP shape.

        A segment can have enough broker-TP proof to preserve its attached-TP
        path while still needing close-discipline repair for MARKET_ORDER_TRADE_CLOSE.
        Thin positive TP buckets are evidence-collection candidates, not proof.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closes = []
            for i in range(20):
                closes.append(
                    {
                        "ts_utc": f"2026-06-02T10:{i:02d}:00Z",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "method": "BREAKOUT_FAILURE",
                        "pl": 500.0,
                    }
                )
            for i in range(3):
                closes.append(
                    {
                        "ts_utc": f"2026-06-02T11:{i:02d}:00Z",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "method": "BREAKOUT_FAILURE",
                        "pl": -1000.0,
                        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                    }
                )
            for i in range(8):
                closes.append(
                    {
                        "ts_utc": f"2026-06-02T12:{i:02d}:00Z",
                        "pair": "GBP_USD",
                        "side": "SHORT",
                        "method": "BREAKOUT_FAILURE",
                        "pl": 450.0,
                    }
                )
            for i in range(3):
                closes.append(
                    {
                        "ts_utc": f"2026-06-02T13:{i:02d}:00Z",
                        "pair": "AUD_JPY",
                        "side": "LONG",
                        "method": "TREND_CONTINUATION",
                        "pl": -700.0,
                        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                    }
                )
            db = root / "ledger.db"
            _make_db(db, closes)
            build_capture_economics(
                ledger_path=db, output_path=root / "out.json", report_path=root / "report.md"
            )
            payload = json.loads((root / "out.json").read_text())
            priorities = payload["segment_repair_priorities"]
            items = {
                (item["pair"], item["side"], item["method"]): item
                for item in priorities["items"]
            }

            eur = items[("EUR_USD", "LONG", "BREAKOUT_FAILURE")]
            self.assertEqual(
                eur["priority_class"],
                "PRESERVE_TP_PROVEN_REPAIR_MARKET_CLOSE_LEAK",
            )
            self.assertTrue(eur["take_profit_proven"])
            self.assertEqual(eur["take_profit_trades"], 20)
            self.assertEqual(eur["take_profit_proof_gap_trades"], 0)
            self.assertEqual(eur["market_close_losses"], 3)
            self.assertEqual(eur["market_close_net_jpy"], -3000.0)
            self.assertEqual(eur["attribution_scope"], "SYSTEM_GATEWAY_ATTRIBUTED_ONLY")
            self.assertTrue(eur["operator_manual_excluded"])
            self.assertTrue(eur["should_count_against_system_edge"])
            self.assertEqual(eur["market_close_loss_trade_ids"], ["t20", "t21", "t22"])
            self.assertEqual(
                eur["market_close_loss_examples"][0]["close_family"],
                "SYSTEM_GATEWAY_MARKET_CLOSE",
            )
            self.assertEqual(
                eur["market_close_loss_examples"][0]["attribution_scope"],
                "SYSTEM_GATEWAY_ATTRIBUTED_ONLY",
            )
            self.assertTrue(
                eur["market_close_loss_examples"][0]["operator_manual_excluded"]
            )

            gbp = items[("GBP_USD", "SHORT", "BREAKOUT_FAILURE")]
            self.assertEqual(gbp["priority_class"], "COLLECT_SCOPED_TP_PROOF")
            self.assertFalse(gbp["take_profit_proven"])
            self.assertEqual(gbp["take_profit_trades"], 8)
            self.assertEqual(gbp["take_profit_proof_gap_trades"], 12)

            aud = items[("AUD_JPY", "LONG", "TREND_CONTINUATION")]
            self.assertEqual(aud["priority_class"], "REPAIR_MARKET_CLOSE_LEAK")
            self.assertEqual(aud["take_profit_trades"], 0)
            self.assertEqual(aud["market_close_net_jpy"], -2100.0)
            report = (root / "report.md").read_text()
            self.assertIn("## Segment Repair Priorities", report)
            self.assertIn("PRESERVE_TP_PROVEN_REPAIR_MARKET_CLOSE_LEAK", report)

    def test_all_wins_sample_reports_positive_expectancy(self) -> None:
        """Zero losses leaves payoff undefined; the verdict must still be
        POSITIVE_EXPECTANCY, not a fall-through NEGATIVE (audit finding)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closes = [
                {"ts_utc": f"2026-06-02T10:{i:02d}:00Z", "pair": "EUR_USD", "pl": 400.0}
                for i in range(21)
            ]
            db = root / "ledger.db"
            _make_db(db, closes)
            summary = build_capture_economics(
                ledger_path=db, output_path=root / "out.json", report_path=root / "report.md"
            )
            self.assertEqual(summary.trades, 21)
            self.assertIsNone(summary.payoff_ratio)
            self.assertEqual(summary.status, "POSITIVE_EXPECTANCY")

    def test_positive_expectancy_and_manual_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closes = []
            for i in range(15):
                closes.append({"ts_utc": f"2026-06-02T10:{i:02d}:00Z", "pair": "EUR_USD", "pl": 1000.0})
            for i in range(10):
                closes.append({"ts_utc": f"2026-06-02T12:{i:02d}:00Z", "pair": "EUR_USD", "pl": -800.0})
            # Manual close must be excluded from the audit entirely.
            closes.append({"ts_utc": "2026-06-02T13:00:00Z", "pair": "USD_JPY", "pl": -99999.0, "attributed": False})
            db = root / "ledger.db"
            _make_db(db, closes)
            summary = build_capture_economics(
                ledger_path=db, output_path=root / "out.json", report_path=root / "report.md"
            )
            self.assertEqual(summary.trades, 25)
            self.assertEqual(summary.status, "POSITIVE_EXPECTANCY")
            payload = json.loads((root / "out.json").read_text())
            self.assertNotIn("USD_JPY", json.dumps(payload["by_exit_reason"]))
            self.assertNotIn("USD_JPY", json.dumps(payload["by_pair_side_exit_reason"]))


if __name__ == "__main__":
    unittest.main()
