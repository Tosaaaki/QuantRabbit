"""Unit tests for capture_economics.py."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import quant_rabbit.capture_economics as capture_module
from quant_rabbit.capture_economics import (
    RealizedOutcome,
    build_capture_economics,
    evaluate_exact_vehicle_net_edge,
    read_attributed_net_outcomes,
    read_attributed_system_entries,
    read_exact_vehicle_allocation_surface,
    read_exact_vehicle_net_metrics,
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
    def test_unresolved_partial_cash_blocks_exact_vehicle_edge_without_increasing_n(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            closes = [
                {
                    "ts_utc": f"2026-07-01T00:{index:02d}:00Z",
                    "pair": "EUR_USD",
                    "pl": 100.0,
                }
                for index in range(20)
            ]
            closes.append(
                {
                    "ts_utc": "2026-07-01T00:30:00Z",
                    "pair": "EUR_USD",
                    "pl": -5000.0,
                    "event_type": "TRADE_REDUCED",
                    "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                }
            )
            _make_db(db, closes)
            financing = {
                "type": "DAILY_FINANCING",
                "financing": "-25.0",
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {"tradeID": "t20", "financing": "-25.0"}
                        ]
                    }
                ],
            }
            with closing(sqlite3.connect(db)) as conn, conn:
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
                        "partial-daily-financing",
                        "2026-07-01T00:31:00Z",
                        -25.0,
                        json.dumps(financing),
                    ),
                )

            metrics = read_exact_vehicle_net_metrics(db)

            self.assertIsNotNone(metrics)
            row = (metrics or {})[
                ("EUR_USD", "LONG", "RANGE_ROTATION", "LIMIT")
            ]
            self.assertEqual(row["trades"], 20)
            self.assertEqual(row["net_jpy"], 2000.0)
            self.assertEqual(row["unresolved_realized_trades"], 1)
            self.assertEqual(row["unresolved_realized_net_jpy"], -5025.0)
            evidence = evaluate_exact_vehicle_net_edge(row)
            self.assertFalse(evidence["proven"])
            self.assertTrue(evidence["blocks_tp_exception"])

    def test_exact_vehicle_edge_requires_arithmetic_identity(self) -> None:
        evidence = evaluate_exact_vehicle_net_edge(
            {
                "trades": 20,
                "wins": 18,
                "losses": 2,
                "net_jpy": 1.0,
                "expectancy_jpy_per_trade": 999.0,
                "avg_win_jpy": 100.0,
                "avg_loss_jpy": 10.0,
                "unresolved_realized_trades": 0,
                "unresolved_realized_net_jpy": 0.0,
            }
        )

        self.assertFalse(evidence["arithmetic_consistent"])
        self.assertFalse(evidence["proven"])
        self.assertTrue(evidence["blocks_tp_exception"])

    def test_thin_tp_exception_never_hides_a_known_all_exit_loss(self) -> None:
        losing_trade = evaluate_exact_vehicle_net_edge(
            {
                "trades": 8,
                "wins": 7,
                "losses": 1,
                "net_jpy": 690.0,
                "expectancy_jpy_per_trade": 86.25,
                "avg_win_jpy": 100.0,
                "avg_loss_jpy": 10.0,
                "unresolved_realized_trades": 0,
                "unresolved_realized_net_jpy": 0.0,
            }
        )
        zero_loss = evaluate_exact_vehicle_net_edge(
            {
                "trades": 8,
                "wins": 8,
                "losses": 0,
                "net_jpy": 800.0,
                "expectancy_jpy_per_trade": 100.0,
                "avg_win_jpy": 100.0,
                "avg_loss_jpy": 0.0,
                "unresolved_realized_trades": 0,
                "unresolved_realized_net_jpy": 0.0,
            }
        )

        self.assertTrue(losing_trade["blocks_tp_exception"])
        self.assertFalse(zero_loss["blocks_tp_exception"])

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

    def test_nanosecond_outcomes_use_shared_legacy_safe_window_parser(self) -> None:
        class LegacyDatetime:
            @staticmethod
            def fromisoformat(value: str) -> datetime:
                timestamp = value.rsplit("+", 1)[0]
                fraction = timestamp.split(".", 1)[1] if "." in timestamp else ""
                if len(fraction) > 6:
                    raise ValueError("legacy runtime rejects nanoseconds")
                return datetime.fromisoformat(value)

        row = RealizedOutcome(
            ts_utc="2026-07-09T03:00:00.123456789Z",
            trade_id="473003",
            pair="EUR_USD",
            side="LONG",
            lane_id="range_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
            method="BREAKOUT_FAILURE",
            exit_reason="TAKE_PROFIT_ORDER",
            realized_pl_jpy=1251.68,
        )
        with patch.object(capture_module, "datetime", LegacyDatetime):
            parsed = capture_module._outcome_timestamp(row)
            week = capture_module._iso_week(row.ts_utc)
            comparison = capture_module._recent_performance_comparison(
                [row],
                as_of=datetime(2026, 7, 10, tzinfo=timezone.utc),
            )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.microsecond if parsed is not None else None, 123456)
        self.assertEqual(week, "2026-W28")
        self.assertEqual(comparison["timestamp_parse_status"], "VALID")
        self.assertEqual(comparison["timestamp_parse_failures"], 0)
        self.assertEqual(comparison["recent_window"]["trades"], 1)

    def test_recent_performance_fails_loud_on_unparseable_lifetime_timestamp(self) -> None:
        row = RealizedOutcome(
            ts_utc="not-a-timestamp",
            trade_id="bad-time",
            pair="EUR_USD",
            side="LONG",
            lane_id="range_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
            method="BREAKOUT_FAILURE",
            exit_reason="TAKE_PROFIT_ORDER",
            realized_pl_jpy=100.0,
        )

        comparison = capture_module._recent_performance_comparison(
            [row],
            as_of=datetime(2026, 7, 10, tzinfo=timezone.utc),
        )

        self.assertEqual(comparison["timestamp_parse_status"], "INVALID")
        self.assertEqual(comparison["timestamp_input_rows"], 1)
        self.assertEqual(comparison["timestamp_parsed_rows"], 0)
        self.assertEqual(comparison["timestamp_parse_failures"], 1)
        self.assertEqual(comparison["verdict"], "TIMESTAMP_PARSE_FAILED")
        self.assertEqual(comparison["proof_status"], "TIMESTAMP_PARSE_FAILED")

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
                    with closing(sqlite3.connect(db)) as conn, conn:
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
                    with closing(sqlite3.connect(db)) as conn, conn:
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
            with closing(sqlite3.connect(db)) as conn, conn:
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
            with closing(sqlite3.connect(db)) as conn, conn:
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
            with closing(sqlite3.connect(db)) as conn, conn:
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
            with closing(sqlite3.connect(db)) as conn, conn:
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
            with closing(sqlite3.connect(db)) as conn, conn:
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
            with closing(sqlite3.connect(db)) as conn, conn:
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

    def test_recent_positive_streak_does_not_clear_negative_lifetime_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closes = []
            for i in range(20):
                closes.append(
                    {
                        "ts_utc": f"2026-06-2{(i % 6) + 1}T10:{i:02d}:00Z",
                        "pair": "EUR_USD",
                        "pl": 200.0 if i < 12 else -900.0,
                        "exit_reason": (
                            "TAKE_PROFIT_ORDER" if i < 12 else "MARKET_ORDER_TRADE_CLOSE"
                        ),
                    }
                )
            for i, pl in enumerate((1251.7, 866.4, 1427.2)):
                closes.append(
                    {
                        "ts_utc": f"2026-07-09T0{i + 3}:00:00Z",
                        "pair": "EUR_USD",
                        "pl": pl,
                        "exit_reason": (
                            "MARKET_ORDER_TRADE_CLOSE" if i == 0 else "TAKE_PROFIT_ORDER"
                        ),
                    }
                )
            db = root / "ledger.db"
            _make_db(db, closes)

            summary = build_capture_economics(
                ledger_path=db,
                output_path=root / "out.json",
                report_path=root / "report.md",
                now=datetime(2026, 7, 11, 13, 0, tzinfo=timezone.utc),
            )

            payload = json.loads((root / "out.json").read_text())
            recent = payload["recent_performance"]
            self.assertEqual(summary.status, "NEGATIVE_EXPECTANCY")
            self.assertEqual(recent["recent_window"]["trades"], 3)
            self.assertEqual(recent["verdict"], "RECENT_POSITIVE_LOW_SAMPLE")
            self.assertFalse(recent["improvement_proven"])
            self.assertEqual(recent["sample_gap"], 17)
            self.assertTrue(recent["baseline_sample_sufficient"])
            self.assertEqual(recent["historical_baseline"]["trades"], 20)
            report = (root / "report.md").read_text()
            self.assertIn("Recent Performance", report)
            self.assertIn("RECENT_POSITIVE_LOW_SAMPLE", report)
            self.assertIn("Improvement proven: `False`", report)

    def test_recent_wins_without_a_historical_baseline_never_prove_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": f"2026-07-{5 + (i // 4):02d}T{i % 24:02d}:00:00Z",
                        "pair": "EUR_USD",
                        "pl": 100.0,
                        "exit_reason": "TAKE_PROFIT_ORDER",
                    }
                    for i in range(20)
                ],
            )

            summary = build_capture_economics(
                ledger_path=db,
                output_path=root / "out.json",
                report_path=root / "report.md",
                now=datetime(2026, 7, 11, 13, 0, tzinfo=timezone.utc),
            )

            recent = json.loads((root / "out.json").read_text())["recent_performance"]
            self.assertEqual(summary.status, "POSITIVE_EXPECTANCY")
            self.assertEqual(recent["recent_window"]["trades"], 20)
            self.assertEqual(recent["verdict"], "POSITIVE_BASELINE_INSUFFICIENT")
            self.assertFalse(recent["baseline_sample_sufficient"])
            self.assertFalse(recent["improvement_proven"])

    def test_raw_jpy_point_improvement_is_observed_but_never_proven(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            historical = [
                {
                    "ts_utc": f"2026-06-{1 + (i // 2):02d}T{i % 24:02d}:00:00Z",
                    "pair": "EUR_USD",
                    "pl": -1.0,
                    "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                }
                for i in range(20)
            ]
            recent_rows = [
                {
                    "ts_utc": f"2026-07-{5 + (i // 4):02d}T{i % 24:02d}:00:00Z",
                    "pair": "EUR_USD",
                    "pl": 0.1,
                    "exit_reason": "TAKE_PROFIT_ORDER",
                }
                for i in range(20)
            ]
            db = root / "ledger.db"
            _make_db(db, [*historical, *recent_rows])

            build_capture_economics(
                ledger_path=db,
                output_path=root / "out.json",
                report_path=root / "report.md",
                now=datetime(2026, 7, 11, 13, 0, tzinfo=timezone.utc),
            )

            recent = json.loads((root / "out.json").read_text())["recent_performance"]
            self.assertEqual(recent["verdict"], "OBSERVED_IMPROVEMENT_MIN_SAMPLE")
            self.assertTrue(recent["observed_improvement"])
            self.assertFalse(recent["improvement_proven"])
            self.assertFalse(recent["normalized_edge_proof_available"])
            self.assertEqual(
                recent["proof_status"],
                "NOT_AVAILABLE_RAW_JPY_POINT_ESTIMATE_ONLY",
            )

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

    def test_open_trade_financing_is_unresolved_and_changes_allocation_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            closes = [
                {
                    "ts_utc": f"2026-07-01T00:{index:02d}:00Z",
                    "pair": "EUR_USD",
                    "pl": 100.0,
                }
                for index in range(20)
            ]
            closes.append(
                {
                    "ts_utc": "2026-07-01T01:00:00Z",
                    "pair": "EUR_USD",
                    "pl": 0.0,
                    "trade_id": "financed-open",
                }
            )
            _make_db(db, closes)
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(
                    "DELETE FROM execution_events WHERE event_type='TRADE_CLOSED' AND trade_id=?",
                    ("financed-open",),
                )
            before = read_exact_vehicle_allocation_surface(db)
            self.assertTrue(
                evaluate_exact_vehicle_net_edge(
                    (read_exact_vehicle_net_metrics(db) or {})[
                        ("EUR_USD", "LONG", "RANGE_ROTATION", "LIMIT")
                    ]
                )["proven"]
            )
            financing = {
                "type": "DAILY_FINANCING",
                "financing": "-5000.0",
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {"tradeID": "financed-open", "financing": "-5000.0"}
                        ]
                    }
                ],
            }
            with closing(sqlite3.connect(db)) as conn, conn:
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
                        "open-daily-financing",
                        "2026-07-01T02:00:00Z",
                        -5000.0,
                        json.dumps(financing),
                    ),
                )

            metrics = read_exact_vehicle_net_metrics(db)
            after = read_exact_vehicle_allocation_surface(db)

            self.assertIsNotNone(metrics)
            row = (metrics or {})[
                ("EUR_USD", "LONG", "RANGE_ROTATION", "LIMIT")
            ]
            self.assertEqual(row["trades"], 20)
            self.assertEqual(row["net_jpy"], 2000.0)
            self.assertEqual(row["unresolved_realized_trades"], 1)
            self.assertEqual(row["unresolved_realized_net_jpy"], -5000.0)
            self.assertEqual(
                row["unresolved_trade_ids_sha256"],
                hashlib.sha256(b'["financed-open"]').hexdigest(),
            )
            self.assertNotEqual(
                before["allocation_surface_sha256"],
                after["allocation_surface_sha256"],
            )
            evidence = evaluate_exact_vehicle_net_edge(row)
            self.assertFalse(evidence["proven"])
            self.assertTrue(evidence["blocks_tp_exception"])

    def test_open_system_financing_with_corrupt_normalized_trade_id_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": "2026-07-01T01:00:00Z",
                        "pair": "EUR_USD",
                        "pl": 0.0,
                        "trade_id": "broken-open",
                    }
                ],
            )
            financing = {
                "id": "broken-financing",
                "type": "DAILY_FINANCING",
                "financing": "-25.0",
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {"tradeID": "broken-open", "financing": "-25.0"}
                        ]
                    }
                ],
            }
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(
                    "DELETE FROM execution_events WHERE event_type='TRADE_CLOSED'"
                )
                conn.execute(
                    "UPDATE execution_events SET trade_id='' WHERE event_type='ORDER_FILLED'"
                )
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
                        "broken-financing-event",
                        "2026-07-01T02:00:00Z",
                        -25.0,
                        json.dumps(financing),
                    ),
                )

            self.assertIsNone(read_attributed_net_outcomes(db))
            self.assertIsNone(read_exact_vehicle_net_metrics(db))
            self.assertEqual(
                read_exact_vehicle_allocation_surface(db)["parse_status"],
                "INVALID",
            )

    def test_authoritative_daily_financing_cannot_disappear_entirely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": "2026-07-01T01:00:00Z",
                        "pair": "EUR_USD",
                        "pl": 0.0,
                        "trade_id": "authoritative-open",
                    }
                ],
            )
            financing = {
                "id": "authoritative-financing",
                "type": "DAILY_FINANCING",
                "financing": "-25.0",
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {
                                "tradeID": "authoritative-open",
                                "financing": "-25.0",
                            }
                        ]
                    }
                ],
            }
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(
                    "DELETE FROM execution_events WHERE event_type='TRADE_CLOSED'"
                )
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
                    "INSERT INTO oanda_transactions VALUES (?, 'DAILY_FINANCING', ?)",
                    ("authoritative-financing", json.dumps(financing)),
                )

            self.assertIsNone(read_attributed_net_outcomes(db))
            self.assertEqual(
                read_exact_vehicle_allocation_surface(db)["parse_status"],
                "INVALID",
            )
            with closing(sqlite3.connect(db)) as conn, conn:
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
                        "authoritative-financing-event",
                        "2026-07-01T02:00:00Z",
                        -25.0,
                        json.dumps(financing),
                    ),
                )

            metrics = read_exact_vehicle_net_metrics(db)
            self.assertIsNotNone(metrics)
            row = (metrics or {})[
                ("EUR_USD", "LONG", "RANGE_ROTATION", "LIMIT")
            ]
            self.assertEqual(row["unresolved_realized_trades"], 1)
            self.assertEqual(row["unresolved_realized_net_jpy"], -25.0)

    def test_idless_financing_cannot_offset_authoritative_broker_cash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": f"2026-07-01T00:{index:02d}:00Z",
                        "pair": "EUR_USD",
                        "pl": 100.0,
                    }
                    for index in range(20)
                ],
            )
            authoritative = {
                "id": "authoritative-financing-loss",
                "type": "DAILY_FINANCING",
                "financing": "-5000.0",
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {"tradeID": "t0", "financing": "-5000.0"}
                        ]
                    }
                ],
            }
            idless_offset = {
                "type": "DAILY_FINANCING",
                "financing": "5000.0",
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {"tradeID": "t0", "financing": "5000.0"}
                        ]
                    }
                ],
            }
            with closing(sqlite3.connect(db)) as conn, conn:
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
                    "INSERT INTO oanda_transactions VALUES (?, 'DAILY_FINANCING', ?)",
                    (
                        "authoritative-financing-loss",
                        json.dumps(authoritative),
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id,
                        trade_id, pair, side, units, realized_pl_jpy,
                        financing_jpy, exit_reason, raw_json
                    ) VALUES (?, ?, 'OANDA_TRANSACTION', NULL, NULL,
                              NULL, NULL, NULL, NULL, NULL, ?, NULL, ?)
                    """,
                    (
                        (
                            "authoritative-financing-loss-event",
                            "2026-07-01T01:00:00Z",
                            -5000.0,
                            json.dumps(authoritative),
                        ),
                        (
                            "idless-financing-offset-event",
                            "2026-07-01T01:01:00Z",
                            5000.0,
                            json.dumps(idless_offset),
                        ),
                    ),
                )

            self.assertIsNone(read_attributed_net_outcomes(db))
            self.assertIsNone(read_exact_vehicle_net_metrics(db))
            self.assertEqual(
                read_exact_vehicle_allocation_surface(db)["parse_status"],
                "INVALID",
            )

    def test_idless_financing_rejected_when_authoritative_table_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            _make_db(
                db,
                [
                    {
                        "ts_utc": f"2026-07-01T00:{index:02d}:00Z",
                        "pair": "EUR_USD",
                        "pl": -100.0,
                        "exit_reason": "STOP_LOSS_ORDER",
                    }
                    for index in range(20)
                ],
            )
            idless_profit = {
                "type": "DAILY_FINANCING",
                "financing": "5000.0",
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {"tradeID": "t0", "financing": "5000.0"}
                        ]
                    }
                ],
            }
            with closing(sqlite3.connect(db)) as conn, conn:
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
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id,
                        trade_id, pair, side, units, realized_pl_jpy,
                        financing_jpy, exit_reason, raw_json
                    ) VALUES (?, ?, 'OANDA_TRANSACTION', NULL, NULL,
                              NULL, NULL, NULL, NULL, NULL, ?, NULL, ?)
                    """,
                    (
                        "idless-financing-profit-event",
                        "2026-07-01T01:00:00Z",
                        5000.0,
                        json.dumps(idless_profit),
                    ),
                )

            self.assertIsNone(read_attributed_net_outcomes(db))
            self.assertIsNone(read_exact_vehicle_net_metrics(db))
            self.assertEqual(
                read_exact_vehicle_allocation_surface(db)["parse_status"],
                "INVALID",
            )

    def test_zero_system_and_manual_open_financing_do_not_change_exact_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            closes = [
                {
                    "ts_utc": f"2026-07-01T00:{index:02d}:00Z",
                    "pair": "EUR_USD",
                    "pl": 100.0,
                }
                for index in range(20)
            ]
            closes.extend(
                [
                    {
                        "ts_utc": "2026-07-01T01:00:00Z",
                        "pair": "EUR_USD",
                        "pl": 0.0,
                        "trade_id": "zero-system-open",
                    },
                    {
                        "ts_utc": "2026-07-01T01:01:00Z",
                        "pair": "EUR_USD",
                        "pl": 0.0,
                        "trade_id": "manual-open",
                        "lane_id": (
                            "operator_manual:EUR_USD:LONG:RANGE_ROTATION:LIMIT"
                        ),
                    },
                ]
            )
            _make_db(db, closes)
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(
                    "DELETE FROM execution_events WHERE event_type='TRADE_CLOSED' "
                    "AND trade_id IN ('zero-system-open', 'manual-open')"
                )
            before = read_exact_vehicle_allocation_surface(db)
            financing = {
                "type": "DAILY_FINANCING",
                "financing": "-25.0",
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {"tradeID": "zero-system-open", "financing": "0.0"},
                            {"tradeID": "manual-open", "financing": "-25.0"},
                        ]
                    }
                ],
            }
            with closing(sqlite3.connect(db)) as conn, conn:
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
                        "zero-manual-financing",
                        "2026-07-01T02:00:00Z",
                        -25.0,
                        json.dumps(financing),
                    ),
                )

            metrics = read_exact_vehicle_net_metrics(db)
            after = read_exact_vehicle_allocation_surface(db)

            self.assertIsNotNone(metrics)
            row = (metrics or {})[
                ("EUR_USD", "LONG", "RANGE_ROTATION", "LIMIT")
            ]
            self.assertEqual(row["trades"], 20)
            self.assertEqual(row["unresolved_realized_trades"], 0)
            self.assertEqual(row["unresolved_realized_net_jpy"], 0.0)
            self.assertEqual(
                row["unresolved_trade_ids_sha256"],
                hashlib.sha256(b"[]").hexdigest(),
            )
            self.assertEqual(
                before["allocation_surface_sha256"],
                after["allocation_surface_sha256"],
            )

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
            with closing(sqlite3.connect(db)) as conn, conn:
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
            with closing(sqlite3.connect(db)) as conn, conn:
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
            with closing(sqlite3.connect(db)) as conn, conn:
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
            with closing(sqlite3.connect(raw_owner_db)) as conn, conn:
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
            with closing(sqlite3.connect(db)) as conn, conn:
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
            with closing(sqlite3.connect(invalid_db)) as conn, conn:
                conn.execute(
                    "UPDATE execution_events SET raw_json='{}' WHERE event_type='TRADE_CLOSED'"
                )
            self.assertIsNone(read_attributed_net_outcomes(invalid_db))

            mismatch_db = root / "mismatch.db"
            _make_db(
                mismatch_db,
                [{"ts_utc": "2026-06-02T10:00:00Z", "pair": "EUR_USD", "pl": 100.0}],
            )
            with closing(sqlite3.connect(mismatch_db)) as conn, conn:
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
            with closing(sqlite3.connect(malformed_db)) as conn, conn:
                conn.execute(
                    "UPDATE execution_events SET raw_json='{bad' WHERE event_type='ORDER_FILLED'"
                )
            self.assertIsNone(read_attributed_net_outcomes(malformed_db))

            cash_db = root / "unsupported-close-cash.db"
            _make_db(
                cash_db,
                [{"ts_utc": "2026-06-02T10:00:00Z", "pair": "EUR_USD", "pl": 100.0}],
            )
            with closing(sqlite3.connect(cash_db)) as conn, conn:
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
            with closing(sqlite3.connect(missing_marker_db)) as conn, conn:
                conn.execute(
                    "DELETE FROM sync_state WHERE key='oanda_transaction_coverage_start_utc'"
                )
            self.assertIsNone(read_attributed_net_outcomes(missing_marker_db))

            precoverage_db = root / "precoverage.db"
            _make_db(
                precoverage_db,
                [{"ts_utc": "2026-06-02T10:00:00Z", "pair": "EUR_USD", "pl": 100.0}],
            )
            with closing(sqlite3.connect(precoverage_db)) as conn, conn:
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

    def test_execution_cost_financing_stress_never_nets_adverse_with_credit(self) -> None:
        outcomes = [
            RealizedOutcome(
                ts_utc=f"2026-07-01T00:{index:02d}:00Z",
                trade_id=f"t{index}",
                pair="EUR_USD",
                side="LONG",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                method="TREND_CONTINUATION",
                exit_reason="TAKE_PROFIT_ORDER",
                realized_pl_jpy=100.0,
                entry_vehicle="MARKET",
                entry_truth_consistent=True,
                broker_close_ts_utc=f"2026-07-01T00:{index:02d}:00Z",
                broker_time_consistent=True,
                entry_units=1000.0,
                # Signed financing is net zero, but the audited component
                # stream contained a 100 JPY debit and a 100 JPY credit.
                audited_financing_jpy=0.0,
                adverse_financing_jpy=100.0 if index == 0 else 0.0,
            )
            for index in range(20)
        ]

        metrics = capture_module._execution_cost_financing_metrics(outcomes)

        self.assertEqual(metrics["adverse_trades"], 1)
        self.assertEqual(metrics["adverse_total_jpy"], 100.0)
        self.assertGreater(metrics["adverse_stress_jpy_per_unit"], 0.0)

    def test_exit_slippage_joins_authoritative_replacement_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.db"
            close_raw = {
                "time": "2026-07-01T00:10:00Z",
                "type": "ORDER_FILL",
                "instrument": "EUR_USD",
                "orderID": "501",
                "reason": "TAKE_PROFIT_ORDER",
                "price": "1.10000",
                "fullVWAP": "1.10000",
                "commission": "0.0",
                "guaranteedExecutionFee": "0.0",
                "tradesClosed": [
                    {
                        "tradeID": "t0",
                        "realizedPL": "100.0",
                        "financing": "0.0",
                        "price": "1.10000",
                    }
                ],
            }
            _make_db(
                db,
                [
                    {
                        "ts_utc": "2026-07-01T00:10:00Z",
                        "entry_ts_utc": "2026-07-01T00:00:00Z",
                        "pair": "EUR_USD",
                        "pl": 100.0,
                        "entry_reason": "LIMIT_ORDER",
                        "exit_reason": "TAKE_PROFIT_ORDER",
                        "raw_json": close_raw,
                    }
                ],
            )
            replaced = {
                "id": "500",
                "type": "TAKE_PROFIT_ORDER",
                "tradeID": "t0",
                "price": "1.09900",
            }
            authoritative = {
                "id": "501",
                "type": "TAKE_PROFIT_ORDER",
                "tradeID": "t0",
                "price": "1.10020",
            }
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(
                    "ALTER TABLE execution_events ADD COLUMN price REAL"
                )
                conn.execute(
                    """
                    UPDATE execution_events
                    SET order_id = '501', price = 1.10000
                    WHERE event_uid = 'c0'
                    """
                )
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id,
                        trade_id, pair, side, units, realized_pl_jpy,
                        financing_jpy, exit_reason, raw_json, price
                    ) VALUES (?, ?, 'PROTECTION_CREATED', NULL, ?, ?, ?, ?,
                              NULL, NULL, 0.0, NULL, ?, NULL)
                    """,
                    (
                        "old-protection",
                        "2026-07-01T00:01:00Z",
                        "500",
                        "t0",
                        "EUR_USD",
                        "LONG",
                        json.dumps(replaced),
                    ),
                )
                conn.execute(
                    """
                    CREATE TABLE oanda_transactions (
                        transaction_id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        raw_json TEXT NOT NULL
                    )
                    """
                )
                conn.executemany(
                    "INSERT INTO oanda_transactions VALUES (?, ?, ?)",
                    (
                        (
                            "500",
                            "TAKE_PROFIT_ORDER",
                            json.dumps(replaced),
                        ),
                        (
                            "501",
                            "TAKE_PROFIT_ORDER",
                            json.dumps(authoritative),
                        ),
                    ),
                )

            surface = capture_module._read_audited_execution_slippage(db)
            self.assertIsNotNone(surface)
            assert surface is not None
            self.assertEqual(surface["take_profit_exit"]["samples"], 1)
            self.assertAlmostEqual(
                surface["take_profit_exit"]["adverse_p95_pips"],
                2.0,
            )

            # The normalized row references the superseded order.  Removing
            # the authoritative replacement must fail closed instead of using
            # that stale trigger or joining merely by trade id.
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(
                    "DELETE FROM oanda_transactions WHERE transaction_id='501'"
                )
            self.assertIsNone(
                capture_module._read_audited_execution_slippage(db)
            )

    def test_exact_financing_cost_uses_key_local_rolling_cohort(self) -> None:
        def outcome(
            index: int,
            closed_at: datetime,
            *,
            adverse_jpy: float = 0.0,
        ) -> RealizedOutcome:
            timestamp = closed_at.isoformat().replace("+00:00", "Z")
            return RealizedOutcome(
                ts_utc=timestamp,
                trade_id=f"rolling-{index}",
                pair="EUR_USD",
                side="LONG",
                lane_id=(
                    "trend_trader:EUR_USD:LONG:"
                    "TREND_CONTINUATION:MARKET"
                ),
                method="TREND_CONTINUATION",
                exit_reason="TAKE_PROFIT_ORDER",
                realized_pl_jpy=100.0,
                entry_vehicle="MARKET",
                entry_truth_consistent=True,
                broker_close_ts_utc=timestamp,
                broker_time_consistent=True,
                entry_units=1000.0,
                audited_financing_jpy=-adverse_jpy,
                adverse_financing_jpy=adverse_jpy,
            )

        latest = datetime(2026, 7, 1, tzinfo=timezone.utc)
        rows = [
            # These lifetime edge rows are outside the exact key's rolling
            # financing cohort and must not dilute the current occurrence
            # rate.  The boundary row exactly 90 days old remains included.
            *[
                outcome(index, latest - timedelta(days=120))
                for index in range(100)
            ],
            outcome(100, latest - timedelta(days=90), adverse_jpy=100.0),
            *[
                outcome(101 + index, latest - timedelta(days=index))
                for index in range(20)
            ],
        ]

        exact = capture_module._aggregate_exact_vehicle_outcomes(
            rows,
            pure_take_profit_only=False,
        )[("EUR_USD", "LONG", "TREND_CONTINUATION", "MARKET")]

        self.assertEqual(exact["trades"], 121)
        self.assertEqual(exact["financing_observation_trades"], 21)
        self.assertEqual(exact["financing_adverse_trades"], 1)
        self.assertEqual(
            exact["financing_oldest_observation_utc"],
            "2026-04-02T00:00:00Z",
        )
        self.assertEqual(
            exact["financing_latest_observation_utc"],
            "2026-07-01T00:00:00Z",
        )
        self.assertGreater(
            exact["financing_adverse_stress_jpy_per_unit"],
            0.0,
        )

    def test_execution_cost_floor_binds_global_and_exact_financing_digest(self) -> None:
        cost_material = {
            "contract": capture_module.EXECUTION_COST_FLOOR_CONTRACT,
            "parse_status": "VALID",
            "scope": "SYSTEM_GATEWAY_ATTRIBUTED_ALL_PAIRS_SIDES_METHODS",
            "minimum_samples": capture_module.EXECUTION_COST_MIN_SAMPLES,
            "maximum_sample_age_seconds": (
                capture_module.EXECUTION_COST_MAX_SAMPLE_AGE_SECONDS
            ),
            "market_entry": {
                "samples": 20,
                "adverse_p95_pips": 0.2,
                "latest_fill_utc": "2026-07-01T00:00:00Z",
                "oldest_fill_utc": "2026-06-01T00:00:00Z",
                "rows_sha256": "1" * 64,
            },
            "take_profit_exit": {
                "samples": 20,
                "adverse_p95_pips": 0.1,
                "latest_fill_utc": "2026-07-01T00:00:00Z",
                "oldest_fill_utc": "2026-06-01T00:00:00Z",
                "rows_sha256": "2" * 64,
            },
            "stop_loss_exit": {
                "samples": 20,
                "adverse_p95_pips": 0.4,
                "latest_fill_utc": "2026-07-01T00:00:00Z",
                "oldest_fill_utc": "2026-06-01T00:00:00Z",
                "rows_sha256": "3" * 64,
            },
            "global_financing": {
                "observation_trades": 20,
                "adverse_trades": 1,
                "adverse_mean_jpy_per_unit": 0.5,
                "adverse_stress_jpy_per_unit": 0.1,
                "latest_observation_utc": "2026-07-01T00:00:00Z",
                "oldest_observation_utc": "2026-06-01T00:00:00Z",
            },
        }
        cost = {
            **cost_material,
            "execution_cost_surface_sha256": (
                capture_module._canonical_json_sha256(cost_material)
            ),
        }
        surface = {
            "parse_status": "VALID",
            "execution_cost": cost,
            "exact_vehicle_net": [
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "method": "TREND_CONTINUATION",
                    "vehicle": "MARKET",
                    "trades": 20,
                    "financing_observation_trades": 20,
                    "financing_adverse_stress_jpy_per_unit": 0.2,
                    "financing_latest_observation_utc": (
                        "2026-07-01T00:00:00Z"
                    ),
                    "financing_oldest_observation_utc": (
                        "2026-06-01T00:00:00Z"
                    ),
                    "latest_broker_close_ts_utc": "2026-07-01T00:00:00Z",
                    "oldest_broker_close_ts_utc": "2026-06-01T00:00:00Z",
                }
            ],
        }
        proof = capture_module.execution_cost_floor_from_surface(
            surface,
            exact_key=("EUR_USD", "LONG", "TREND_CONTINUATION", "MARKET"),
            as_of=datetime(2026, 7, 2, tzinfo=timezone.utc),
        )
        self.assertEqual(proof["status"], "PASSED")
        self.assertEqual(proof["financing_adverse_stress_jpy_per_unit"], 0.2)
        tampered = json.loads(json.dumps(surface))
        tampered["execution_cost"]["market_entry"]["adverse_p95_pips"] = 0.0
        blocked = capture_module.execution_cost_floor_from_surface(
            tampered,
            exact_key=("EUR_USD", "LONG", "TREND_CONTINUATION", "MARKET"),
            as_of=datetime(2026, 7, 2, tzinfo=timezone.utc),
        )
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIn(
            "EXECUTION_COST_SURFACE_DIGEST_MISMATCH",
            blocked["failed_checks"],
        )

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
