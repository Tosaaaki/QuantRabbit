"""Unit tests for strategy/daily_review.py."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.strategy.daily_review import (
    DAILY_REVIEW_MAX_BIAS,
    DAILY_REVIEW_N_LOSSES_FOR_BLOCK,
    DAILY_REVIEW_N_TRADES_FOR_BIAS,
    compute_daily_review,
    write_trader_overrides,
)


def _make_db(path: Path, trades: list[dict]) -> None:
    """Minimal execution_events table with realized outcome rows."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE execution_events (
            event_uid TEXT PRIMARY KEY,
            ts_utc TEXT NOT NULL,
            source TEXT NOT NULL,
            event_type TEXT NOT NULL,
            lane_id TEXT,
            trade_id TEXT,
            pair TEXT,
            side TEXT,
            units INTEGER,
            realized_pl_jpy REAL,
            related_transaction_ids_json TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            inserted_at_utc TEXT NOT NULL
        )
        """
    )
    for i, t in enumerate(trades):
        conn.execute(
            """
            INSERT INTO execution_events
                (event_uid, ts_utc, source, event_type, lane_id, trade_id, pair, side,
                 units, realized_pl_jpy, related_transaction_ids_json, raw_json, inserted_at_utc)
            VALUES (?, ?, 'test', ?, ?, ?, ?, ?, ?, ?, '[]', '{}', ?)
            """,
            (
                f"uid-{i}",
                t["ts_utc"],
                t.get("event_type", "TRADE_CLOSED"),
                t.get("lane_id"),
                t.get("trade_id", f"trade-{i}"),
                t["pair"],
                t.get("close_side", t.get("side")),
                t.get("units"),
                t["pl"],
                t["ts_utc"],
            ),
        )
    conn.commit()
    conn.close()


class DailyReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)

    def _ts(self, hours_back: float) -> str:
        return (self.now - timedelta(hours=hours_back)).isoformat().replace("+00:00", "Z")

    def test_missing_db_returns_empty_report(self) -> None:
        report = compute_daily_review(Path("/nonexistent.db"), now=self.now)
        self.assertEqual(report.bias_overrides, {})
        self.assertEqual(report.blocked_lanes, [])
        self.assertIsNotNone(report.expires_at_utc)

    def test_empty_window_returns_empty_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(path, [])
            report = compute_daily_review(path, now=self.now)
            self.assertEqual(report.bias_overrides, {})

    def test_losing_direction_triggers_negative_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            # 3 closes for original GBP_USD LONG (close_side=SHORT), net loss
            _make_db(path, [
                {"pair": "GBP_USD", "close_side": "SHORT", "pl": -1500, "ts_utc": self._ts(20)},
                {"pair": "GBP_USD", "close_side": "SHORT", "pl": -2000, "ts_utc": self._ts(15)},
                {"pair": "GBP_USD", "close_side": "SHORT", "pl": -1200, "ts_utc": self._ts(10)},
            ])
            report = compute_daily_review(path, now=self.now)
            self.assertIn("GBP_USD", report.bias_overrides)
            self.assertIn("LONG", report.bias_overrides["GBP_USD"])
            delta = report.bias_overrides["GBP_USD"]["LONG"]
            self.assertLess(delta, 0)
            self.assertGreaterEqual(delta, -DAILY_REVIEW_MAX_BIAS)

    def test_winning_direction_triggers_positive_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(path, [
                {"pair": "USD_JPY", "close_side": "SHORT", "pl": +2500, "ts_utc": self._ts(20)},
                {"pair": "USD_JPY", "close_side": "SHORT", "pl": +1800, "ts_utc": self._ts(15)},
                {"pair": "USD_JPY", "close_side": "SHORT", "pl": +1500, "ts_utc": self._ts(10)},
            ])
            report = compute_daily_review(path, now=self.now)
            self.assertIn("USD_JPY", report.bias_overrides)
            self.assertGreater(report.bias_overrides["USD_JPY"]["LONG"], 0)

    def test_fewer_than_min_trades_no_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            # Only 2 trades, below N_TRADES_FOR_BIAS = 3
            _make_db(path, [
                {"pair": "EUR_USD", "close_side": "LONG", "pl": -5000, "ts_utc": self._ts(20)},
                {"pair": "EUR_USD", "close_side": "LONG", "pl": -3000, "ts_utc": self._ts(10)},
            ])
            report = compute_daily_review(path, now=self.now)
            self.assertNotIn("EUR_USD", report.bias_overrides)

    def test_small_net_pl_does_not_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            # 3 trades but net -300 (below 1000 threshold)
            _make_db(path, [
                {"pair": "EUR_JPY", "close_side": "LONG", "pl": -100, "ts_utc": self._ts(20)},
                {"pair": "EUR_JPY", "close_side": "LONG", "pl": -100, "ts_utc": self._ts(15)},
                {"pair": "EUR_JPY", "close_side": "LONG", "pl": -100, "ts_utc": self._ts(10)},
            ])
            report = compute_daily_review(path, now=self.now)
            self.assertNotIn("EUR_JPY", report.bias_overrides)

    def test_window_filters_old_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            # 3 recent + 3 old (>24h)
            _make_db(path, [
                {"pair": "AUD_JPY", "close_side": "SHORT", "pl": -2000, "ts_utc": self._ts(48)},
                {"pair": "AUD_JPY", "close_side": "SHORT", "pl": -2000, "ts_utc": self._ts(40)},
                {"pair": "AUD_JPY", "close_side": "SHORT", "pl": -2000, "ts_utc": self._ts(30)},
                {"pair": "EUR_USD", "close_side": "SHORT", "pl": -2000, "ts_utc": self._ts(20)},
                {"pair": "EUR_USD", "close_side": "SHORT", "pl": -2000, "ts_utc": self._ts(15)},
                {"pair": "EUR_USD", "close_side": "SHORT", "pl": -2000, "ts_utc": self._ts(10)},
            ])
            report = compute_daily_review(path, now=self.now, lookback_hours=24)
            # AUD_JPY trades are >24h old, should NOT appear
            self.assertNotIn("AUD_JPY", report.bias_overrides)
            # EUR_USD trades are recent, should appear
            self.assertIn("EUR_USD", report.bias_overrides)

    def test_lane_id_with_n_losses_gets_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            trades = [
                {"pair": "GBP_USD", "close_side": "SHORT", "pl": -1000,
                 "ts_utc": self._ts(20 - i), "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT"}
                for i in range(DAILY_REVIEW_N_LOSSES_FOR_BLOCK)
            ]
            _make_db(path, trades)
            report = compute_daily_review(path, now=self.now)
            self.assertIn("failure_trader:GBP_USD:LONG:BREAKOUT", report.blocked_lanes)

    def test_lane_id_with_fewer_losses_not_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            # Just 1 loss for this lane → not blocked
            _make_db(path, [
                {"pair": "GBP_USD", "close_side": "SHORT", "pl": -1000,
                 "ts_utc": self._ts(10), "lane_id": "lane-a"},
            ])
            report = compute_daily_review(path, now=self.now)
            self.assertEqual(report.blocked_lanes, [])

    def test_partial_reductions_collapse_to_one_trade_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            lane_id = "range_trader:AUD_JPY:SHORT:RANGE_ROTATION"
            _make_db(path, [
                {
                    "pair": "AUD_JPY",
                    "side": "SHORT",
                    "units": -9700,
                    "pl": None,
                    "event_type": "ORDER_FILLED",
                    "trade_id": "471020",
                    "ts_utc": self._ts(20),
                    "lane_id": lane_id,
                },
                {
                    "pair": "AUD_JPY",
                    "close_side": "LONG",
                    "units": 6500,
                    "pl": -2515.5,
                    "event_type": "TRADE_REDUCED",
                    "trade_id": "471020",
                    "ts_utc": self._ts(19),
                    "lane_id": lane_id,
                },
                {
                    "pair": "AUD_JPY",
                    "close_side": "LONG",
                    "units": 3200,
                    "pl": -1276.8,
                    "event_type": "TRADE_REDUCED",
                    "trade_id": "471020",
                    "ts_utc": self._ts(18),
                    "lane_id": lane_id,
                },
                {
                    "pair": "AUD_JPY",
                    "close_side": "LONG",
                    "units": 3300,
                    "pl": -155.1,
                    "event_type": "TRADE_CLOSED",
                    "trade_id": "471020",
                    "ts_utc": self._ts(17),
                    "lane_id": lane_id,
                },
            ])

            report = compute_daily_review(path, now=self.now)

            self.assertEqual(report.lane_loss_counts[lane_id], 1)
            self.assertEqual(report.blocked_lanes, [])
            self.assertNotIn("AUD_JPY", report.bias_overrides)
            self.assertAlmostEqual(report.pair_pl_breakdown[("AUD_JPY", "SHORT")], -3947.4)

    def test_expiry_is_next_jst_midnight(self) -> None:
        # now = 2026-05-13 12:00 UTC = 21:00 JST May 13
        # next JST midnight = 2026-05-14 00:00 JST = 2026-05-13 15:00 UTC
        report = compute_daily_review(Path("/nonexistent.db"), now=self.now)
        self.assertEqual(report.expires_at_utc, "2026-05-13T15:00:00Z")

    def test_write_trader_overrides_produces_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ledger.db"
            out_path = Path(tmp) / "trader_overrides.json"
            _make_db(db_path, [
                {"pair": "GBP_USD", "close_side": "SHORT", "pl": -1500, "ts_utc": self._ts(20)},
                {"pair": "GBP_USD", "close_side": "SHORT", "pl": -2000, "ts_utc": self._ts(15)},
                {"pair": "GBP_USD", "close_side": "SHORT", "pl": -1200, "ts_utc": self._ts(10)},
            ])
            report = compute_daily_review(db_path, now=self.now)
            write_trader_overrides(report, out_path)
            self.assertTrue(out_path.exists())
            data = json.loads(out_path.read_text())
            self.assertIn("expires_at_utc", data)
            self.assertIn("bias_overrides", data)
            self.assertIn("GBP_USD", data["bias_overrides"])


if __name__ == "__main__":
    unittest.main()
