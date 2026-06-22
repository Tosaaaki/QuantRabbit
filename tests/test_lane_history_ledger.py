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
    compute_same_day_lane_loss_streaks,
    compute_same_day_loss_streaks,
    lane_history_modifier,
)


def _make_db(path: Path, trades: list[dict]) -> None:
    """Write a minimal execution_events table with realized outcome rows."""
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
    entry_trade_ids = {
        t.get("trade_id", f"trade-{i}")
        for i, t in enumerate(trades)
        if t.get("event_type", "TRADE_CLOSED") == "ORDER_FILLED"
    }
    expanded: list[dict] = []
    for i, t in enumerate(trades):
        trade_id = t.get("trade_id", f"trade-{i}")
        event_type = t.get("event_type", "TRADE_CLOSED")
        lane_id = t.get("lane_id", f"test_lane:{t['pair']}:{_original_side(t['side'])}:TEST")
        if event_type == "ORDER_FILLED" and t.get("attributed", True):
            expanded.append(
                {
                    "event_uid": f"gateway-{i}",
                    "ts_utc": t["ts_utc"],
                    "event_type": "GATEWAY_ORDER_SENT",
                    "lane_id": lane_id,
                    "order_id": f"order-{trade_id}",
                    "trade_id": trade_id,
                    "pair": t["pair"],
                    "side": t["side"],
                    "units": t.get("units"),
                    "pl": None,
                }
            )
        if event_type in {"TRADE_CLOSED", "TRADE_REDUCED"} and t.get("attributed", True) and trade_id not in entry_trade_ids:
            original = _original_side(t["side"])
            expanded.extend(
                [
                    {
                        "event_uid": f"gateway-{i}",
                        "ts_utc": t["ts_utc"],
                        "event_type": "GATEWAY_ORDER_SENT",
                        "lane_id": lane_id,
                        "order_id": f"order-{trade_id}",
                        "trade_id": trade_id,
                        "pair": t["pair"],
                        "side": original,
                        "units": _units_for_side(original),
                        "pl": None,
                    },
                    {
                        "event_uid": f"entry-{i}",
                        "ts_utc": t["ts_utc"],
                        "event_type": "ORDER_FILLED",
                        "lane_id": None,
                        "order_id": f"order-{trade_id}",
                        "trade_id": trade_id,
                        "pair": t["pair"],
                        "side": original,
                        "units": _units_for_side(original),
                        "pl": None,
                    },
                ]
            )
        expanded.append(
            {
                "event_uid": f"uid-{i}",
                "ts_utc": t["ts_utc"],
                "event_type": event_type,
                "lane_id": t.get("lane_id"),
                "order_id": t.get("order_id", f"order-{trade_id}" if event_type == "ORDER_FILLED" else f"close-{trade_id}"),
                "trade_id": trade_id,
                "pair": t["pair"],
                "side": t["side"],
                "units": t.get("units"),
                "pl": t["pl"],
            }
        )
    for row in expanded:
        conn.execute(
            """
            INSERT INTO execution_events
                (event_uid, ts_utc, source, event_type, trade_id, pair, side,
                 lane_id, order_id, units, realized_pl_jpy, related_transaction_ids_json, raw_json, inserted_at_utc)
            VALUES (?, ?, 'test', ?, ?, ?, ?, ?, ?, ?, ?, '[]', '{}', ?)
            """,
            (
                row["event_uid"],
                row["ts_utc"],
                row["event_type"],
                row["trade_id"],
                row["pair"],
                row["side"],
                row.get("lane_id"),
                row.get("order_id"),
                row.get("units"),
                row["pl"],
                row["ts_utc"],
            ),
        )
    conn.commit()
    conn.close()


def _original_side(close_side: str) -> str:
    side = str(close_side).upper()
    if side == "LONG":
        return "SHORT"
    if side == "SHORT":
        return "LONG"
    return side


def _units_for_side(side: str) -> int:
    return 1000 if str(side).upper() == "LONG" else -1000


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

    def test_partial_reductions_are_aggregated_by_original_position_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {
                        "pair": "AUD_JPY",
                        "side": "SHORT",
                        "units": -9700,
                        "pl": None,
                        "event_type": "ORDER_FILLED",
                        "trade_id": "471020",
                        "ts_utc": "2026-05-14T01:00:00Z",
                    },
                    {
                        "pair": "AUD_JPY",
                        "side": "LONG",
                        "units": 6500,
                        "pl": -2515.5,
                        "event_type": "TRADE_REDUCED",
                        "trade_id": "471020",
                        "ts_utc": "2026-05-14T04:55:17Z",
                    },
                    {
                        "pair": "AUD_JPY",
                        "side": "LONG",
                        "units": 3200,
                        "pl": -1276.8,
                        "event_type": "TRADE_REDUCED",
                        "trade_id": "471020",
                        "ts_utc": "2026-05-14T05:37:54Z",
                    },
                    {
                        "pair": "AUD_JPY",
                        "side": "LONG",
                        "units": 3300,
                        "pl": -155.1,
                        "event_type": "TRADE_CLOSED",
                        "trade_id": "471020",
                        "ts_utc": "2026-05-14T16:20:47Z",
                    },
                ],
            )

            hist = compute_lane_history(path)

            snap = hist[("AUD_JPY", "SHORT")]
            self.assertEqual(snap.sample_size, 1)
            self.assertAlmostEqual(snap.net_pl_jpy, -3947.4)
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

    def test_method_specific_history_overrides_pair_direction_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "pl": -2000.0,
                        "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                        "ts_utc": "2026-05-12T00:00:00Z",
                    },
                    {
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "pl": -1500.0,
                        "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                        "ts_utc": "2026-05-13T00:00:00Z",
                    },
                    {
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "pl": +2000.0,
                        "lane_id": "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION",
                        "ts_utc": "2026-05-14T00:00:00Z",
                    },
                    {
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "pl": +1500.0,
                        "lane_id": "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION",
                        "ts_utc": "2026-05-15T00:00:00Z",
                    },
                ],
            )

            hist = compute_lane_history(path)
            range_delta, range_rationale = lane_history_modifier(
                hist,
                "EUR_USD",
                "SHORT",
                "RANGE_ROTATION",
            )
            trend_delta, trend_rationale = lane_history_modifier(
                hist,
                "EUR_USD",
                "SHORT",
                "TREND_CONTINUATION",
            )
            fallback_delta, _ = lane_history_modifier(hist, "EUR_USD", "SHORT")

        self.assertLess(range_delta, 0.0)
        self.assertIn("EUR_USD:SHORT:RANGE_ROTATION", range_rationale or "")
        self.assertGreater(trend_delta, 0.0)
        self.assertIn("EUR_USD:SHORT:TREND_CONTINUATION", trend_rationale or "")
        self.assertAlmostEqual(fallback_delta, 0.0)

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

    def test_skips_non_realized_events(self) -> None:
        """Only realized rows contribute; ORDER_FILLED / others ignored.

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

    def test_ignores_unattributed_manual_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "pl": +22725.0,
                        "ts_utc": "2026-05-07T22:25:28Z",
                        "attributed": False,
                    },
                    {
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "pl": -500.0,
                        "ts_utc": "2026-05-08T01:00:00Z",
                    },
                ],
            )

            hist = compute_lane_history(path)

            snap = hist[("USD_JPY", "LONG")]
            self.assertEqual(snap.sample_size, 1)
            self.assertAlmostEqual(snap.net_pl_jpy, -500.0)


class SameDayLossStreakTest(unittest.TestCase):
    """compute_same_day_loss_streaks — 2026-06-04 EUR_USD regression model."""

    def test_missing_db_returns_empty(self) -> None:
        self.assertEqual(compute_same_day_loss_streaks(Path("/nonexistent.db"), "2026-06-04"), {})

    def test_consecutive_losses_pool_both_directions(self) -> None:
        """LONG loss, LONG loss, revenge-flip SHORT loss → streak 3 on the pair."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    # Closing LONG issues SELL → close side SHORT.
                    {"pair": "EUR_USD", "side": "SHORT", "pl": -2181.0, "ts_utc": "2026-06-04T11:30:00Z", "trade_id": "t1"},
                    {"pair": "EUR_USD", "side": "SHORT", "pl": -2642.0, "ts_utc": "2026-06-04T12:04:00Z", "trade_id": "t2"},
                    # Closing SHORT issues BUY → close side LONG.
                    {"pair": "EUR_USD", "side": "LONG", "pl": -2333.0, "ts_utc": "2026-06-04T14:44:00Z", "trade_id": "t3"},
                ],
            )
            streaks = compute_same_day_loss_streaks(path, "2026-06-04")
            self.assertIn("EUR_USD", streaks)
            self.assertEqual(streaks["EUR_USD"].consecutive_losses, 3)
            self.assertAlmostEqual(streaks["EUR_USD"].net_loss_jpy, -7156.0)
            self.assertEqual(streaks["EUR_USD"].last_loss_ts_utc, "2026-06-04T14:44:00Z")

    def test_winning_close_resets_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {"pair": "EUR_USD", "side": "SHORT", "pl": -2181.0, "ts_utc": "2026-06-04T11:30:00Z", "trade_id": "t1"},
                    {"pair": "EUR_USD", "side": "SHORT", "pl": +900.0, "ts_utc": "2026-06-04T12:30:00Z", "trade_id": "t2"},
                    {"pair": "EUR_USD", "side": "SHORT", "pl": -500.0, "ts_utc": "2026-06-04T13:30:00Z", "trade_id": "t3"},
                ],
            )
            streaks = compute_same_day_loss_streaks(path, "2026-06-04")
            self.assertEqual(streaks["EUR_USD"].consecutive_losses, 1)
            self.assertAlmostEqual(streaks["EUR_USD"].net_loss_jpy, -500.0)

    def test_other_day_losses_do_not_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {"pair": "EUR_USD", "side": "SHORT", "pl": -3307.0, "ts_utc": "2026-05-29T02:18:00Z", "trade_id": "t1"},
                    {"pair": "EUR_USD", "side": "SHORT", "pl": -5267.0, "ts_utc": "2026-05-29T02:34:00Z", "trade_id": "t2"},
                ],
            )
            self.assertEqual(compute_same_day_loss_streaks(path, "2026-06-04"), {})
            self.assertEqual(
                compute_same_day_loss_streaks(path, "2026-05-29")["EUR_USD"].consecutive_losses, 2
            )

    def test_unattributed_manual_close_does_not_count(self) -> None:
        """Manual/tagless closes must not gate the bot (manual P/L separation)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {"pair": "USD_JPY", "side": "SHORT", "pl": -4000.0, "ts_utc": "2026-06-04T10:00:00Z", "trade_id": "m1", "attributed": False},
                    {"pair": "EUR_USD", "side": "SHORT", "pl": -2181.0, "ts_utc": "2026-06-04T11:30:00Z", "trade_id": "t1"},
                ],
            )
            streaks = compute_same_day_loss_streaks(path, "2026-06-04")
            self.assertNotIn("USD_JPY", streaks)
            self.assertIn("EUR_USD", streaks)

    def test_breakeven_close_neither_counts_nor_resets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {"pair": "EUR_USD", "side": "SHORT", "pl": -2181.0, "ts_utc": "2026-06-04T11:30:00Z", "trade_id": "t1"},
                    {"pair": "EUR_USD", "side": "SHORT", "pl": 0.0, "ts_utc": "2026-06-04T12:00:00Z", "trade_id": "t2"},
                    {"pair": "EUR_USD", "side": "SHORT", "pl": -100.0, "ts_utc": "2026-06-04T12:30:00Z", "trade_id": "t3"},
                ],
            )
            streaks = compute_same_day_loss_streaks(path, "2026-06-04")
            self.assertEqual(streaks["EUR_USD"].consecutive_losses, 2)


class SameDayLaneLossStreakTest(unittest.TestCase):
    """compute_same_day_lane_loss_streaks — P0 repair loop breaker."""

    def test_exact_pair_side_method_loss_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "pl": -280.8,
                        "ts_utc": "2026-06-22T05:00:00Z",
                        "trade_id": "t1",
                        "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                    }
                ],
            )

            streaks = compute_same_day_lane_loss_streaks(path, "2026-06-22")

            key = ("EUR_USD", "LONG", "RANGE_ROTATION")
            self.assertIn(key, streaks)
            self.assertEqual(streaks[key].consecutive_losses, 1)
            self.assertAlmostEqual(streaks[key].net_loss_jpy, -280.8)

    def test_exact_lane_win_resets_loss_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "pl": -280.8,
                        "ts_utc": "2026-06-22T05:00:00Z",
                        "trade_id": "t1",
                        "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                    },
                    {
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "pl": +385.0,
                        "ts_utc": "2026-06-22T05:30:00Z",
                        "trade_id": "t2",
                        "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                    },
                ],
            )

            streaks = compute_same_day_lane_loss_streaks(path, "2026-06-22")

            self.assertNotIn(("EUR_USD", "LONG", "RANGE_ROTATION"), streaks)

    def test_different_method_keeps_separate_streaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "pl": -280.8,
                        "ts_utc": "2026-06-22T05:00:00Z",
                        "trade_id": "t1",
                        "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                    },
                    {
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "pl": -100.0,
                        "ts_utc": "2026-06-22T05:30:00Z",
                        "trade_id": "t2",
                        "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                    },
                ],
            )

            streaks = compute_same_day_lane_loss_streaks(path, "2026-06-22")

            self.assertIn(("EUR_USD", "LONG", "RANGE_ROTATION"), streaks)
            self.assertIn(("EUR_USD", "LONG", "BREAKOUT_FAILURE"), streaks)


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
