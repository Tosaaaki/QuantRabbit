from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.execution_timing_contracts import (
    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
    TP_PROGRESS_REPAIR_REPLAY_FIELD,
)
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

            self.assertEqual(
                payload["precision"][TP_PROGRESS_REPAIR_REPLAY_FIELD],
                TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
            )
            summary = payload["summary"]
            self.assertEqual(summary["canceled_orders_audited"], 1)
            self.assertEqual(summary["canceled_entry_touched_after_cancel"], 1)
            self.assertEqual(summary["canceled_entry_touched_after_cancel_rate"], 1.0)
            self.assertEqual(summary["canceled_positive_after_cancel_entry"], 1)
            self.assertEqual(summary["canceled_positive_after_cancel_entry_rate"], 1.0)
            self.assertEqual(summary["canceled_tp_touched_after_cancel"], 1)
            self.assertEqual(summary["canceled_tp_touched_after_cancel_rate"], 1.0)
            self.assertAlmostEqual(summary["canceled_estimated_missed_mfe_jpy"], 120.0)
            row = payload["canceled_order_regrets"][0]
            self.assertEqual(row["entry_touch_after_cancel_minutes"], 0.5)
            self.assertEqual(row["tp_touch_after_cancel_minutes"], 9.5)
            self.assertEqual(row["mfe_pips_after_cancel_entry"], 12.0)

    def test_canceled_order_regret_rolls_up_by_pair_side_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(db)
            with sqlite3.connect(db) as conn:
                for idx, cancel_minute in enumerate((10, 12), start=1):
                    order_id = f"o{idx}"
                    _insert_event(
                        conn,
                        f"accepted-{order_id}",
                        ts_utc=f"2026-06-16T00:0{idx}:00Z",
                        event_type="ORDER_ACCEPTED",
                        lane_id="failure_trader:USD_JPY:LONG:BREAKOUT_FAILURE",
                        order_id=order_id,
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
                        f"canceled-{order_id}",
                        ts_utc=f"2026-06-16T00:{cancel_minute}:30Z",
                        event_type="ORDER_CANCELED",
                        order_id=order_id,
                        pair="USD_JPY",
                    )
                conn.commit()

            candles = (
                BidAskCandle(_dt("2026-06-16T00:13:00Z"), 150.02, 149.98, 150.03, 149.99),
                BidAskCandle(_dt("2026-06-16T00:20:00Z"), 150.12, 150.05, 150.13, 150.06),
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

            rollup = payload["canceled_order_regret_by_shape"]
            self.assertEqual(rollup["total_shapes"], 1)
            item = rollup["items"][0]
            self.assertEqual(
                item["evidence_ref"],
                "timing:canceled_shape:USD_JPY:LONG:BREAKOUT_FAILURE:LIMIT_ORDER",
            )
            self.assertEqual(item["priority_class"], "PRESERVE_PENDING_THESIS_TP_TOUCHED")
            self.assertEqual(item["orders"], 2)
            self.assertEqual(item["entry_touched_after_cancel"], 2)
            self.assertEqual(item["tp_touched_after_cancel"], 2)
            self.assertEqual(item["entry_touch_after_cancel_rate"], 1.0)
            self.assertEqual(item["tp_touched_after_cancel_rate"], 1.0)
            self.assertAlmostEqual(item["estimated_missed_mfe_jpy"], 240.0)
            report = (root / "audit.md").read_text()
            self.assertIn("## Canceled Order Regret By Shape", report)
            self.assertIn("PRESERVE_PENDING_THESIS_TP_TOUCHED", report)

    def test_canceled_order_window_is_clamped_to_now(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(db)
            with sqlite3.connect(db) as conn:
                _insert_event(
                    conn,
                    "accepted-o1",
                    ts_utc="2026-06-16T23:45:00Z",
                    event_type="ORDER_ACCEPTED",
                    lane_id="lane:recent-cancel",
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
                    ts_utc="2026-06-16T23:55:30Z",
                    event_type="ORDER_CANCELED",
                    order_id="o1",
                    pair="USD_JPY",
                )
                conn.commit()

            now = _dt("2026-06-17T00:00:00Z")
            requested: list[tuple[datetime, datetime]] = []

            def fetcher(pair: str, start: datetime, end: datetime, granularity: str) -> tuple[BidAskCandle, ...]:
                requested.append((start, end))
                self.assertLessEqual(end, now)
                return (
                    BidAskCandle(_dt("2026-06-16T23:56:00Z"), 150.02, 149.98, 150.03, 149.99),
                )

            payload = build_execution_timing_audit(
                ledger_path=db,
                snapshot_path=None,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
                now_utc=now,
                candle_fetcher=fetcher,
            )

            self.assertTrue(requested)
            self.assertEqual(payload["status"], "OK")
            self.assertEqual(payload["summary"]["canceled_orders_audited"], 1)

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
            self.assertEqual(summary["loss_closes_had_positive_mfe_rate"], 1.0)
            self.assertEqual(summary["loss_closes_tp_touched_before_close"], 1)
            self.assertEqual(summary["loss_closes_tp_touched_before_close_rate"], 1.0)
            self.assertAlmostEqual(summary["loss_close_estimated_mfe_jpy"], 110.0)
            self.assertAlmostEqual(summary["loss_close_actual_pl_jpy"], -100.0)
            self.assertAlmostEqual(summary["loss_close_counterfactual_profit_capture_pl_jpy"], 100.0)
            self.assertAlmostEqual(summary["loss_close_counterfactual_profit_capture_delta_jpy"], 200.0)
            self.assertAlmostEqual(summary["loss_close_counterfactual_profit_capture_jpy"], 100.0)
            self.assertEqual(summary["avg_decision_lag_minutes_after_first_positive"], 5.5)
            row = payload["loss_close_regrets"][0]
            self.assertEqual(row["first_positive_minutes_after_fill"], 0.83)
            self.assertEqual(row["decision_lag_minutes_after_first_positive"], 5.5)
            self.assertEqual(row["mfe_pips_before_loss_close"], 11.0)
            self.assertEqual(row["profit_capture_counterfactual_exit"], "TAKE_PROFIT_TOUCH")
            self.assertEqual(row["profit_capture_counterfactual_pips"], 10.0)
            self.assertAlmostEqual(row["profit_capture_counterfactual_jpy"], 100.0)
            self.assertAlmostEqual(row["profit_capture_counterfactual_net_improvement_jpy"], 200.0)

    def test_stop_loss_close_reports_missed_tp_progress_profit_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(db)
            with sqlite3.connect(db) as conn:
                _insert_event(
                    conn,
                    "fill-t-sl",
                    ts_utc="2026-06-16T01:00:10Z",
                    event_type="ORDER_FILLED",
                    lane_id="range_trader:USD_JPY:SHORT:RANGE_ROTATION",
                    order_id="o-fill",
                    trade_id="t-sl",
                    pair="USD_JPY",
                    side="SHORT",
                    units=-1000,
                    price=150.00,
                )
                _insert_event(
                    conn,
                    "tp-t-sl",
                    ts_utc="2026-06-16T01:00:20Z",
                    event_type="PROTECTION_CREATED",
                    trade_id="t-sl",
                    pair="USD_JPY",
                    price=149.94,
                    raw={"type": "TAKE_PROFIT_ORDER", "tradeID": "t-sl", "price": "149.94"},
                )
                _insert_event(
                    conn,
                    "sl-t-sl",
                    ts_utc="2026-06-16T01:00:20Z",
                    event_type="PROTECTION_CREATED",
                    trade_id="t-sl",
                    pair="USD_JPY",
                    price=150.05,
                    raw={"type": "STOP_LOSS_ORDER", "tradeID": "t-sl", "price": "150.05"},
                )
                _insert_event(
                    conn,
                    "close-t-sl",
                    ts_utc="2026-06-16T01:10:30Z",
                    event_type="TRADE_CLOSED",
                    order_id="o-sl",
                    trade_id="t-sl",
                    pair="USD_JPY",
                    side="SHORT",
                    units=1000,
                    price=150.05,
                    realized_pl_jpy=-50.0,
                    exit_reason="STOP_LOSS_ORDER",
                )
                conn.commit()

            candles = (
                BidAskCandle(_dt("2026-06-16T01:03:00Z"), 149.958, 149.948, 149.960, 149.952),
                BidAskCandle(_dt("2026-06-16T01:08:00Z"), 150.030, 149.990, 150.050, 150.010),
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
            self.assertEqual(summary["loss_closes_profit_capture_missed"], 1)
            self.assertEqual(summary["stop_loss_closes_profit_capture_missed"], 1)
            self.assertAlmostEqual(summary["loss_close_estimated_capture_gap_jpy"], 48.0)
            row = payload["loss_close_regrets"][0]
            self.assertEqual(row["exit_reason"], "STOP_LOSS_ORDER")
            self.assertEqual(row["mfe_pips_before_loss_close"], 4.8)
            self.assertAlmostEqual(row["tp_progress_before_loss_close"], 0.8)
            self.assertTrue(row["profit_capture_missed_before_loss_close"])

    def test_stop_loss_close_marks_thirty_percent_tp_progress_capture_missed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(db)
            with sqlite3.connect(db) as conn:
                _insert_event(
                    conn,
                    "fill-t-sl-30",
                    ts_utc="2026-06-16T01:00:10Z",
                    event_type="ORDER_FILLED",
                    lane_id="range_trader:USD_JPY:SHORT:RANGE_ROTATION",
                    order_id="o-fill-30",
                    trade_id="t-sl-30",
                    pair="USD_JPY",
                    side="SHORT",
                    units=-1000,
                    price=150.00,
                )
                _insert_event(
                    conn,
                    "tp-t-sl-30",
                    ts_utc="2026-06-16T01:00:20Z",
                    event_type="PROTECTION_CREATED",
                    trade_id="t-sl-30",
                    pair="USD_JPY",
                    price=149.90,
                    raw={"type": "TAKE_PROFIT_ORDER", "tradeID": "t-sl-30", "price": "149.90"},
                )
                _insert_event(
                    conn,
                    "sl-t-sl-30",
                    ts_utc="2026-06-16T01:00:20Z",
                    event_type="PROTECTION_CREATED",
                    trade_id="t-sl-30",
                    pair="USD_JPY",
                    price=150.05,
                    raw={"type": "STOP_LOSS_ORDER", "tradeID": "t-sl-30", "price": "150.05"},
                )
                _insert_event(
                    conn,
                    "close-t-sl-30",
                    ts_utc="2026-06-16T01:10:30Z",
                    event_type="TRADE_CLOSED",
                    order_id="o-sl-30",
                    trade_id="t-sl-30",
                    pair="USD_JPY",
                    side="SHORT",
                    units=1000,
                    price=150.05,
                    realized_pl_jpy=-50.0,
                    exit_reason="STOP_LOSS_ORDER",
                )
                conn.commit()

            candles = (
                BidAskCandle(_dt("2026-06-16T01:03:00Z"), 149.970, 149.960, 149.975, 149.968),
                BidAskCandle(_dt("2026-06-16T01:08:00Z"), 150.030, 149.990, 150.050, 150.010),
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
            self.assertEqual(summary["loss_closes_profit_capture_missed"], 1)
            self.assertEqual(summary["stop_loss_closes_profit_capture_missed"], 1)
            self.assertAlmostEqual(summary["loss_close_estimated_capture_gap_jpy"], 32.0)
            self.assertAlmostEqual(summary["loss_close_actual_pl_jpy"], -50.0)
            self.assertAlmostEqual(summary["loss_close_counterfactual_profit_capture_pl_jpy"], 30.0)
            self.assertAlmostEqual(summary["loss_close_counterfactual_profit_capture_delta_jpy"], 80.0)
            self.assertAlmostEqual(summary["loss_close_counterfactual_profit_capture_jpy"], 30.0)
            self.assertEqual(summary["loss_closes_repair_replay_triggered"], 1)
            self.assertAlmostEqual(summary["loss_close_repair_replay_profit_capture_jpy"], 32.0)
            self.assertAlmostEqual(summary["loss_close_repair_replay_delta_jpy"], 82.0)
            row = payload["loss_close_regrets"][0]
            self.assertEqual(row["mfe_pips_before_loss_close"], 3.2)
            self.assertAlmostEqual(row["tp_progress_before_loss_close"], 0.32)
            self.assertTrue(row["profit_capture_missed_before_loss_close"])
            self.assertEqual(row["profit_capture_counterfactual_exit"], "TP_PROGRESS_CAPTURE")
            self.assertEqual(row["profit_capture_counterfactual_pips"], 3.0)
            self.assertAlmostEqual(row["profit_capture_counterfactual_jpy"], 30.0)
            self.assertAlmostEqual(row["profit_capture_counterfactual_net_improvement_jpy"], 80.0)
            self.assertTrue(row["repair_replay_triggered_before_loss_close"])
            self.assertEqual(row["repair_replay_exit"], "TP_PROGRESS_PRODUCTION_GATE_REPLAY")
            self.assertAlmostEqual(row["repair_replay_profit_pips"], 3.2)
            self.assertAlmostEqual(row["repair_replay_counterfactual_jpy"], 32.0)
            self.assertAlmostEqual(row["repair_replay_counterfactual_net_improvement_jpy"], 82.0)

    def test_tp_progress_repair_replay_requires_live_noise_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(db)
            with sqlite3.connect(db) as conn:
                _insert_event(
                    conn,
                    "fill-noisy",
                    ts_utc="2026-06-16T01:00:10Z",
                    event_type="ORDER_FILLED",
                    lane_id="range_trader:USD_JPY:SHORT:RANGE_ROTATION",
                    order_id="o-fill-noisy",
                    trade_id="t-noisy",
                    pair="USD_JPY",
                    side="SHORT",
                    units=-1000,
                    price=150.00,
                )
                _insert_event(
                    conn,
                    "tp-noisy",
                    ts_utc="2026-06-16T01:00:20Z",
                    event_type="PROTECTION_CREATED",
                    trade_id="t-noisy",
                    pair="USD_JPY",
                    price=149.90,
                    raw={"type": "TAKE_PROFIT_ORDER", "tradeID": "t-noisy", "price": "149.90"},
                )
                _insert_event(
                    conn,
                    "close-noisy",
                    ts_utc="2026-06-16T01:10:30Z",
                    event_type="TRADE_CLOSED",
                    order_id="o-close-noisy",
                    trade_id="t-noisy",
                    pair="USD_JPY",
                    side="SHORT",
                    units=1000,
                    price=150.05,
                    realized_pl_jpy=-50.0,
                    exit_reason="STOP_LOSS_ORDER",
                )
                conn.commit()

            candles = (
                BidAskCandle(_dt("2026-06-16T01:03:00Z"), 150.080, 149.960, 150.090, 149.968),
                BidAskCandle(_dt("2026-06-16T01:08:00Z"), 150.030, 149.990, 150.050, 150.010),
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
            self.assertEqual(summary["loss_closes_profit_capture_missed"], 1)
            self.assertEqual(summary["loss_closes_repair_replay_triggered"], 0)
            self.assertAlmostEqual(summary["loss_close_repair_replay_counterfactual_pl_jpy"], -50.0)
            row = payload["loss_close_regrets"][0]
            self.assertTrue(row["profit_capture_missed_before_loss_close"])
            self.assertFalse(row["repair_replay_triggered_before_loss_close"])

    def test_market_close_lookback_is_anchored_to_latest_close_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(db)
            with sqlite3.connect(db) as conn:
                _insert_event(
                    conn,
                    "fill-old-loss",
                    ts_utc="2026-06-15T03:00:00Z",
                    event_type="ORDER_FILLED",
                    lane_id="lane:old-loss",
                    order_id="o-old-fill",
                    trade_id="t-old-loss",
                    pair="USD_JPY",
                    side="LONG",
                    units=1000,
                    price=150.00,
                )
                _insert_event(
                    conn,
                    "close-old-loss",
                    ts_utc="2026-06-15T03:55:58Z",
                    event_type="TRADE_CLOSED",
                    order_id="o-old-close",
                    trade_id="t-old-loss",
                    pair="USD_JPY",
                    side="LONG",
                    units=1000,
                    price=149.90,
                    realized_pl_jpy=-100.0,
                    exit_reason="MARKET_ORDER_TRADE_CLOSE",
                )
                _insert_event(
                    conn,
                    "fill-latest",
                    ts_utc="2026-06-19T14:00:00Z",
                    event_type="ORDER_FILLED",
                    lane_id="lane:latest",
                    order_id="o-latest-fill",
                    trade_id="t-latest",
                    pair="USD_JPY",
                    side="LONG",
                    units=1000,
                    price=150.00,
                )
                _insert_event(
                    conn,
                    "close-latest",
                    ts_utc="2026-06-19T14:22:09Z",
                    event_type="TRADE_CLOSED",
                    order_id="o-latest-close",
                    trade_id="t-latest",
                    pair="USD_JPY",
                    side="LONG",
                    units=1000,
                    price=150.03,
                    realized_pl_jpy=30.0,
                    exit_reason="MARKET_ORDER_TRADE_CLOSE",
                )
                conn.commit()

            candles = (
                BidAskCandle(_dt("2026-06-15T03:01:00Z"), 150.02, 149.99, 150.03, 150.00),
                BidAskCandle(_dt("2026-06-15T03:56:00Z"), 150.08, 149.89, 150.09, 149.90),
                BidAskCandle(_dt("2026-06-19T14:23:00Z"), 150.06, 150.01, 150.07, 150.02),
            )

            def fetcher(pair: str, start: datetime, end: datetime, granularity: str) -> tuple[BidAskCandle, ...]:
                self.assertEqual(pair, "USD_JPY")
                return tuple(c for c in candles if start <= c.timestamp_utc <= end)

            payload = build_execution_timing_audit(
                ledger_path=db,
                snapshot_path=None,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
                now_utc=_dt("2026-06-22T05:07:51Z"),
                candle_fetcher=fetcher,
            )

            loss_rows = {row["trade_id"]: row for row in payload["loss_close_regrets"]}
            market_rows = {row["trade_id"]: row for row in payload["market_close_counterfactuals"]}
            self.assertIn("t-old-loss", loss_rows)
            self.assertIn("t-old-loss", market_rows)
            self.assertEqual(payload["summary"]["loss_market_closes_audited"], 1)
            self.assertEqual(market_rows["t-old-loss"]["post_close_path_label"], "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE")
            self.assertEqual(
                payload["window"]["market_close_from_utc"],
                "2026-06-12T14:22:09+00:00",
            )

    def test_market_close_counterfactual_splits_followthrough_from_risk_containment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.db"
            _make_db(db)
            with sqlite3.connect(db) as conn:
                _insert_event(
                    conn,
                    "fill-profit",
                    ts_utc="2026-06-16T01:00:00Z",
                    event_type="ORDER_FILLED",
                    lane_id="lane:profit-runner",
                    order_id="o-profit",
                    trade_id="t-profit",
                    pair="USD_JPY",
                    side="LONG",
                    units=1000,
                    price=150.00,
                )
                _insert_event(
                    conn,
                    "tp-profit",
                    ts_utc="2026-06-16T01:00:10Z",
                    event_type="PROTECTION_CREATED",
                    trade_id="t-profit",
                    pair="USD_JPY",
                    price=150.14,
                    raw={"type": "TAKE_PROFIT_ORDER", "tradeID": "t-profit", "price": "150.14"},
                )
                _insert_event(
                    conn,
                    "gateway-profit",
                    ts_utc="2026-06-16T01:10:00Z",
                    event_type="GATEWAY_TRADE_CLOSE_SENT",
                    trade_id="t-profit",
                    pair="USD_JPY",
                    exit_reason="TAKE_PROFIT_MARKET",
                )
                _insert_event(
                    conn,
                    "close-profit",
                    ts_utc="2026-06-16T01:10:05Z",
                    event_type="TRADE_CLOSED",
                    order_id="close-profit",
                    trade_id="t-profit",
                    pair="USD_JPY",
                    side="LONG",
                    units=1000,
                    price=150.05,
                    realized_pl_jpy=50.0,
                    exit_reason="MARKET_ORDER_TRADE_CLOSE",
                )
                _insert_event(
                    conn,
                    "fill-loss",
                    ts_utc="2026-06-16T02:00:00Z",
                    event_type="ORDER_FILLED",
                    lane_id="lane:loss-cut",
                    order_id="o-loss",
                    trade_id="t-loss",
                    pair="USD_JPY",
                    side="SHORT",
                    units=1000,
                    price=150.00,
                )
                _insert_event(
                    conn,
                    "sl-loss",
                    ts_utc="2026-06-16T02:00:10Z",
                    event_type="PROTECTION_CREATED",
                    trade_id="t-loss",
                    pair="USD_JPY",
                    price=150.18,
                    raw={"type": "STOP_LOSS_ORDER", "tradeID": "t-loss", "price": "150.18"},
                )
                _insert_event(
                    conn,
                    "gateway-loss",
                    ts_utc="2026-06-16T02:10:00Z",
                    event_type="GATEWAY_TRADE_CLOSE_SENT",
                    trade_id="t-loss",
                    pair="USD_JPY",
                    exit_reason="GPT_CLOSE",
                )
                _insert_event(
                    conn,
                    "close-loss",
                    ts_utc="2026-06-16T02:10:05Z",
                    event_type="TRADE_CLOSED",
                    order_id="close-loss",
                    trade_id="t-loss",
                    pair="USD_JPY",
                    side="SHORT",
                    units=1000,
                    price=150.08,
                    realized_pl_jpy=-80.0,
                    exit_reason="MARKET_ORDER_TRADE_CLOSE",
                )
                conn.commit()

            candles = (
                BidAskCandle(_dt("2026-06-16T01:11:00Z"), 150.16, 150.04, 150.17, 150.05),
                BidAskCandle(_dt("2026-06-16T02:11:00Z"), 150.19, 150.02, 150.20, 150.03),
            )

            def fetcher(pair: str, start: datetime, end: datetime, granularity: str) -> tuple[BidAskCandle, ...]:
                self.assertEqual(pair, "USD_JPY")
                return tuple(c for c in candles if start <= c.timestamp_utc <= end)

            payload = build_execution_timing_audit(
                ledger_path=db,
                snapshot_path=None,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
                now_utc=_dt("2026-06-16T03:00:00Z"),
                candle_fetcher=fetcher,
            )

            summary = payload["summary"]
            self.assertEqual(summary["market_closes_audited"], 2)
            self.assertEqual(summary["profit_market_closes_left_runner_upside"], 1)
            self.assertEqual(summary["loss_market_closes_contained_risk"], 1)
            self.assertEqual(summary["market_close_estimated_followthrough_jpy"], 140.0)
            self.assertEqual(summary["market_close_estimated_avoided_adverse_jpy"], 120.0)
            rows = {row["trade_id"]: row for row in payload["market_close_counterfactuals"]}
            self.assertEqual(rows["t-profit"]["gateway_action"], "TAKE_PROFIT_MARKET")
            self.assertEqual(rows["t-profit"]["post_close_path_label"], "PROFIT_CLOSE_LEFT_RUNNER_UPSIDE")
            self.assertTrue(rows["t-profit"]["tp_touched_after_market_close"])
            self.assertEqual(rows["t-loss"]["gateway_action"], "GPT_CLOSE")
            self.assertEqual(rows["t-loss"]["post_close_path_label"], "LOSS_CLOSE_CONTAINED_RISK")
            self.assertTrue(rows["t-loss"]["sl_touched_after_market_close"])


if __name__ == "__main__":
    unittest.main()
