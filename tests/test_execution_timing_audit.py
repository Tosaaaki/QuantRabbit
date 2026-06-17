from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.execution_timing_audit import BidAskCandle, build_execution_timing_audit


def _make_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                source TEXT,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                client_order_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                tp REAL,
                sl REAL,
                realized_pl_jpy REAL,
                financing_jpy REAL,
                exit_reason TEXT,
                oanda_transaction_id TEXT,
                related_transaction_ids_json TEXT,
                raw_json TEXT,
                inserted_at_utc TEXT
            )
            """
        )
        conn.commit()


def _insert_event(conn: sqlite3.Connection, event_uid: str, **values: Any) -> None:
    payload = {
        "event_uid": event_uid,
        "ts_utc": values.pop("ts_utc"),
        "source": "test",
        "event_type": values.pop("event_type"),
        "lane_id": values.pop("lane_id", None),
        "order_id": values.pop("order_id", None),
        "trade_id": values.pop("trade_id", None),
        "client_order_id": values.pop("client_order_id", None),
        "pair": values.pop("pair", None),
        "side": values.pop("side", None),
        "units": values.pop("units", None),
        "price": values.pop("price", None),
        "tp": values.pop("tp", None),
        "sl": values.pop("sl", None),
        "realized_pl_jpy": values.pop("realized_pl_jpy", None),
        "financing_jpy": values.pop("financing_jpy", None),
        "exit_reason": values.pop("exit_reason", None),
        "oanda_transaction_id": values.pop("oanda_transaction_id", None),
        "related_transaction_ids_json": json.dumps(values.pop("related_transaction_ids", [])),
        "raw_json": json.dumps(values.pop("raw", {})),
        "inserted_at_utc": values.pop("inserted_at_utc", "2026-06-17T00:00:00Z"),
    }
    if values:
        raise AssertionError(f"unused event fields: {sorted(values)}")
    conn.execute(
        """
        INSERT INTO execution_events (
            event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id,
            client_order_id, pair, side, units, price, tp, sl, realized_pl_jpy,
            financing_jpy, exit_reason, oanda_transaction_id,
            related_transaction_ids_json, raw_json, inserted_at_utc
        ) VALUES (
            :event_uid, :ts_utc, :source, :event_type, :lane_id, :order_id, :trade_id,
            :client_order_id, :pair, :side, :units, :price, :tp, :sl, :realized_pl_jpy,
            :financing_jpy, :exit_reason, :oanda_transaction_id,
            :related_transaction_ids_json, :raw_json, :inserted_at_utc
        )
        """,
        payload,
    )


def _dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


class ExecutionTimingAuditTest(unittest.TestCase):
    def test_canceled_limit_order_reports_missed_tp_and_mfe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(db)
            with sqlite3.connect(db) as conn:
                _insert_event(
                    conn,
                    "accepted-o1",
                    ts_utc="2026-06-16T00:00:00Z",
                    event_type="ORDER_ACCEPTED",
                    lane_id="lane:missed",
                    order_id="o1",
                    pair="USD_JPY",
                    side="LONG",
                    units=1000,
                    price=150.00,
                    tp=150.10,
                    sl=149.80,
                    exit_reason="CLIENT_ORDER",
                    raw={
                        "type": "LIMIT_ORDER",
                        "instrument": "USD_JPY",
                        "units": "1000",
                        "price": "150.00",
                        "takeProfitOnFill": {"price": "150.10"},
                        "stopLossOnFill": {"price": "149.80"},
                    },
                )
                _insert_event(
                    conn,
                    "canceled-o1",
                    ts_utc="2026-06-16T00:10:30Z",
                    event_type="ORDER_CANCELED",
                    order_id="o1",
                    pair="USD_JPY",
                )
                conn.commit()

            candles = (
                BidAskCandle(_dt("2026-06-16T00:11:00Z"), 150.02, 149.98, 150.03, 149.99),
                BidAskCandle(_dt("2026-06-16T00:20:00Z"), 150.12, 150.05, 150.13, 150.06),
            )

            def fetcher(pair: str, start: datetime, end: datetime, granularity: str) -> tuple[BidAskCandle, ...]:
                self.assertEqual(pair, "USD_JPY")
                self.assertEqual(granularity, "M1")
                return tuple(c for c in candles if start <= c.timestamp_utc <= end)

            payload = build_execution_timing_audit(
                ledger_path=db,
                snapshot_path=None,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
                now_utc=_dt("2026-06-17T00:00:00Z"),
                candle_fetcher=fetcher,
            )

            summary = payload["summary"]
            self.assertEqual(summary["canceled_orders_audited"], 1)
            self.assertEqual(summary["canceled_entry_touched_after_cancel"], 1)
            self.assertEqual(summary["canceled_positive_after_cancel_entry"], 1)
            self.assertEqual(summary["canceled_tp_touched_after_cancel"], 1)
            self.assertAlmostEqual(summary["canceled_estimated_missed_mfe_jpy"], 120.0)
            row = payload["canceled_order_regrets"][0]
            self.assertEqual(row["entry_touch_after_cancel_minutes"], 0.5)
            self.assertEqual(row["tp_touch_after_cancel_minutes"], 9.5)
            self.assertEqual(row["mfe_pips_after_cancel_entry"], 12.0)

    def test_losing_close_reports_prior_positive_mfe_and_decision_lag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(db)
            with sqlite3.connect(db) as conn:
                _insert_event(
                    conn,
                    "fill-t1",
                    ts_utc="2026-06-16T01:00:10Z",
                    event_type="ORDER_FILLED",
                    lane_id="lane:late-exit",
                    order_id="o-fill",
                    trade_id="t1",
                    pair="USD_JPY",
                    side="LONG",
                    units=1000,
                    price=150.00,
                )
                _insert_event(
                    conn,
                    "tp-t1",
                    ts_utc="2026-06-16T01:00:20Z",
                    event_type="PROTECTION_CREATED",
                    trade_id="t1",
                    pair="USD_JPY",
                    price=150.10,
                    raw={"type": "TAKE_PROFIT_ORDER", "tradeID": "t1", "price": "150.10"},
                )
                _insert_event(
                    conn,
                    "close-t1",
                    ts_utc="2026-06-16T01:06:30Z",
                    event_type="TRADE_CLOSED",
                    order_id="o-close",
                    trade_id="t1",
                    pair="USD_JPY",
                    side="LONG",
                    units=1000,
                    price=149.90,
                    realized_pl_jpy=-100.0,
                    exit_reason="MARKET_ORDER_TRADE_CLOSE",
                )
                conn.commit()

            candles = (
                BidAskCandle(_dt("2026-06-16T01:01:00Z"), 150.02, 149.99, 150.03, 150.00),
                BidAskCandle(_dt("2026-06-16T01:03:00Z"), 150.11, 150.04, 150.12, 150.05),
                BidAskCandle(_dt("2026-06-16T01:05:00Z"), 150.00, 149.93, 150.01, 149.94),
            )

            def fetcher(pair: str, start: datetime, end: datetime, granularity: str) -> tuple[BidAskCandle, ...]:
                self.assertEqual(pair, "USD_JPY")
                return tuple(c for c in candles if start <= c.timestamp_utc <= end)

            payload = build_execution_timing_audit(
                ledger_path=db,
                snapshot_path=None,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
                now_utc=_dt("2026-06-17T00:00:00Z"),
                candle_fetcher=fetcher,
            )

            summary = payload["summary"]
            self.assertEqual(summary["loss_closes_audited"], 1)
            self.assertEqual(summary["loss_closes_had_positive_mfe"], 1)
            self.assertEqual(summary["loss_closes_tp_touched_before_close"], 1)
            self.assertAlmostEqual(summary["loss_close_estimated_mfe_jpy"], 110.0)
            self.assertEqual(summary["avg_decision_lag_minutes_after_first_positive"], 5.5)
            row = payload["loss_close_regrets"][0]
            self.assertEqual(row["first_positive_minutes_after_fill"], 0.83)
            self.assertEqual(row["decision_lag_minutes_after_first_positive"], 5.5)
            self.assertEqual(row["mfe_pips_before_loss_close"], 11.0)


if __name__ == "__main__":
    unittest.main()
