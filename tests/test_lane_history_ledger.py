"""Unit tests for strategy/lane_history_ledger.py."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.strategy.lane_history_ledger import (
    LANE_HISTORY_LOOKBACK_TRADES,
    LANE_HISTORY_MAX_MODIFIER,
    LANE_HISTORY_SATURATION_PL_JPY,
    LaneHistorySnapshot,
    compute_lane_history,
    lane_history_modifier,
)


def _make_db(path: Path, trades: list[dict]) -> None:
    """Write a minimal execution_events table with TRADE_CLOSED rows."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE execution_events (
            event_uid TEXT PRIMARY KEY,
            ts_utc TEXT NOT NULL,
            source TEXT NOT NULL,
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
                (event_uid, ts_utc, source, event_type, pair, side,
                 realized_pl_jpy, related_transaction_ids_json, raw_json, inserted_at_utc)
            VALUES (?, ?, 'test', ?, ?, ?, ?, '[]', '{}', ?)
            """,
            (
                f"uid-{i}",
                t["ts_utc"],
                t.get("event_type", "TRADE_CLOSED"),
                t["pair"],
                t["side"],
                t["pl"],
                t["ts_utc"],
            ),
        )
    conn.commit()
    conn.close()


class LaneHistoryLedgerTest(unittest.TestCase):
    def test_missing_db_returns_empty(self) -> None:
        self.assertEqual(compute_lane_history(Path("/nonexistent.db")), {})

    def test_empty_db_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(path, [])
            self.assertEqual(compute_lane_history(path), {})

    def test_close_fill_side_is_inverted(self) -> None:
        """TRADE_CLOSED.side records the closing fill direction. Original
        position direction is the opposite — verify the inversion."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    # Closing a LONG position issues a SELL → side=SHORT
                    {"pair": "GBP_USD", "side": "SHORT", "pl": -1000.0, "ts_utc": "2026-05-12T15:00:00Z"},
                    # Closing a SHORT position issues a BUY → side=LONG
                    {"pair": "AUD_JPY", "side": "LONG", "pl": +500.0, "ts_utc": "2026-05-12T16:00:00Z"},
                ],
            )
            hist = compute_lane_history(path)
            # The lane key is the ORIGINAL position direction.
            self.assertIn(("GBP_USD", "LONG"), hist)
            self.assertIn(("AUD_JPY", "SHORT"), hist)
            self.assertNotIn(("GBP_USD", "SHORT"), hist)
            self.assertNotIn(("AUD_JPY", "LONG"), hist)

    def test_modifier_negative_for_losing_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            losses = [
                {"pair": "EUR_USD", "side": "SHORT", "pl": -2000.0, "ts_utc": f"2026-05-1{i}T00:00:00Z"}
                for i in range(2, 7)
            ]
            _make_db(path, losses)
            hist = compute_lane_history(path)
            snap = hist[("EUR_USD", "LONG")]  # closes were SHORT → original LONG
            self.assertEqual(snap.sample_size, LANE_HISTORY_LOOKBACK_TRADES)
            self.assertLess(snap.modifier, -20.0)
            self.assertLessEqual(snap.modifier, 0.0)

    def test_modifier_positive_for_winning_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            wins = [
                {"pair": "EUR_USD", "side": "LONG", "pl": +2000.0, "ts_utc": f"2026-05-1{i}T00:00:00Z"}
                for i in range(2, 7)
            ]
            _make_db(path, wins)
            hist = compute_lane_history(path)
            snap = hist[("EUR_USD", "SHORT")]  # closes were LONG → original SHORT
            self.assertGreater(snap.modifier, +20.0)

    def test_modifier_bounded_by_max(self) -> None:
        """Even extreme losses cap at ±LANE_HISTORY_MAX_MODIFIER."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            heavy_losses = [
                {"pair": "USD_JPY", "side": "LONG", "pl": -50000.0, "ts_utc": f"2026-05-1{i}T00:00:00Z"}
                for i in range(2, 7)
            ]
            _make_db(path, heavy_losses)
            hist = compute_lane_history(path)
            snap = hist[("USD_JPY", "SHORT")]
            self.assertAlmostEqual(snap.modifier, -LANE_HISTORY_MAX_MODIFIER, places=2)

    def test_only_last_N_trades_aggregated(self) -> None:
        """LANE_HISTORY_LOOKBACK_TRADES sets the rolling window."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            # Older big wins...
            older = [
                {"pair": "EUR_JPY", "side": "SHORT", "pl": +10000.0, "ts_utc": f"2026-05-0{i}T00:00:00Z"}
                for i in range(1, 4)
            ]
            # ...followed by very recent big losses.
            newer = [
                {"pair": "EUR_JPY", "side": "SHORT", "pl": -10000.0, "ts_utc": f"2026-05-1{i}T00:00:00Z"}
                for i in range(0, 5)
            ]
            _make_db(path, older + newer)
            hist = compute_lane_history(path)
            snap = hist[("EUR_JPY", "LONG")]
            # Only the most recent 5 should count → modifier near saturation negative
            self.assertEqual(snap.sample_size, LANE_HISTORY_LOOKBACK_TRADES)
            self.assertLess(snap.modifier, -20.0)

    def test_skips_rows_with_null_pl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {"pair": "EUR_USD", "side": "SHORT", "pl": None, "ts_utc": "2026-05-12T00:00:00Z"},
                    {"pair": "EUR_USD", "side": "SHORT", "pl": -500.0, "ts_utc": "2026-05-13T00:00:00Z"},
                ],
            )
            hist = compute_lane_history(path)
            snap = hist[("EUR_USD", "LONG")]
            self.assertEqual(snap.sample_size, 1)
            self.assertAlmostEqual(snap.net_pl_jpy, -500.0)

    def test_skips_non_closed_events(self) -> None:
        """Only TRADE_CLOSED rows contribute; ORDER_FILLED / others ignored.

        The ORDER_FILLED LONG +1000 should NOT show up. The TRADE_CLOSED
        with close_side=SHORT inverts to original LONG, net -500.
        Sample size must be 1, proving the ORDER_FILLED row was skipped.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {"pair": "EUR_USD", "side": "LONG", "pl": +1000.0, "event_type": "ORDER_FILLED", "ts_utc": "2026-05-12T00:00:00Z"},
                    {"pair": "EUR_USD", "side": "SHORT", "pl": -500.0, "event_type": "TRADE_CLOSED", "ts_utc": "2026-05-13T00:00:00Z"},
                ],
            )
            hist = compute_lane_history(path)
            snap = hist.get(("EUR_USD", "LONG"))
            self.assertIsNotNone(snap)
            self.assertEqual(snap.sample_size, 1)
            self.assertAlmostEqual(snap.net_pl_jpy, -500.0)


class LaneHistoryModifierLookupTest(unittest.TestCase):
    def test_missing_lane_returns_zero(self) -> None:
        delta, rationale = lane_history_modifier({}, "EUR_USD", "LONG")
        self.assertEqual(delta, 0.0)
        self.assertIsNone(rationale)

    def test_long_lookup_finds_long_snapshot(self) -> None:
        snap = LaneHistorySnapshot(
            pair="EUR_USD", direction="LONG", sample_size=3,
            net_pl_jpy=-5000.0, modifier=-20.0,
        )
        snapshots = {("EUR_USD", "LONG"): snap}
        delta, rationale = lane_history_modifier(snapshots, "EUR_USD", "LONG")
        self.assertAlmostEqual(delta, -20.0)
        self.assertIn("EUR_USD:LONG", rationale or "")
        self.assertIn("-5000", rationale or "")


if __name__ == "__main__":
    unittest.main()
