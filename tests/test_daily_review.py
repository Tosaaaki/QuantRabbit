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
    DAILY_REVIEW_STRUCTURAL_N_TRADES_FOR_BIAS,
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
            order_id TEXT,
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
    entry_trade_ids = {
        t.get("trade_id", f"trade-{i}")
        for i, t in enumerate(trades)
        if t.get("event_type", "TRADE_CLOSED") == "ORDER_FILLED"
    }
    expanded: list[dict] = []
    for i, t in enumerate(trades):
        trade_id = t.get("trade_id", f"trade-{i}")
        event_type = t.get("event_type", "TRADE_CLOSED")
        close_side = t.get("close_side", t.get("side"))
        original = _original_side(close_side)
        entry_lane_id = t.get("entry_lane_id", t.get("lane_id", f"test_lane:{t['pair']}:{original}:TEST"))
        if event_type in {"TRADE_CLOSED", "TRADE_REDUCED"} and t.get("attributed", True) and trade_id not in entry_trade_ids:
            expanded.extend(
                [
                    {
                        "event_uid": f"gateway-{i}",
                        "ts_utc": t["ts_utc"],
                        "event_type": "GATEWAY_ORDER_SENT",
                        "lane_id": entry_lane_id,
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
                "side": close_side,
                "units": t.get("units"),
                "pl": t["pl"],
            }
        )
    for row in expanded:
        conn.execute(
            """
            INSERT INTO execution_events
                (event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id, pair, side,
                 units, realized_pl_jpy, related_transaction_ids_json, raw_json, inserted_at_utc)
            VALUES (?, ?, 'test', ?, ?, ?, ?, ?, ?, ?, ?, '[]', '{}', ?)
            """,
            (
                row["event_uid"],
                row["ts_utc"],
                row["event_type"],
                row.get("lane_id"),
                row.get("order_id"),
                row["trade_id"],
                row["pair"],
                row["side"],
                row.get("units"),
                row["pl"],
                row["ts_utc"],
            ),
        )
    conn.commit()
    conn.close()


def _make_target_path_review_db(path: Path, rows: list[dict]) -> None:
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
            pair TEXT,
            side TEXT,
            units INTEGER,
            realized_pl_jpy REAL,
            exit_reason TEXT,
            related_transaction_ids_json TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            inserted_at_utc TEXT NOT NULL
        )
        """
    )
    for row in rows:
        conn.execute(
            """
            INSERT INTO execution_events
                (event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id, pair, side,
                 units, realized_pl_jpy, exit_reason, related_transaction_ids_json, raw_json, inserted_at_utc)
            VALUES (?, ?, 'test', ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', ?, ?)
            """,
            (
                row["event_uid"],
                row["ts_utc"],
                row["event_type"],
                row.get("lane_id"),
                row.get("order_id"),
                row.get("trade_id"),
                row.get("pair"),
                row.get("side"),
                row.get("units"),
                row.get("realized_pl_jpy"),
                row.get("exit_reason"),
                json.dumps(row.get("raw_json", {}), sort_keys=True),
                row["ts_utc"],
            ),
        )
    conn.commit()
    conn.close()


def _target_path_sent_row(*, ts_utc: str, trade_id: str, lane_id: str = "target:EUR_USD:LONG:HERO") -> dict:
    receipt = {
        "daily_target_mode": "ATTACK",
        "remaining_to_5pct": 3000.0,
        "five_pct_path_role": "HERO",
        "attack_stack_slot": "NOW",
        "grade": "A",
        "suggested_units": 1000,
        "final_units": 1000,
        "risk_yen": 900.0,
        "risk_pct": 0.45,
        "target_yen": 1500.0,
        "contribution_to_5pct": 1500.0,
        "live_order_gateway_receipt_id": "qrv1-EURUSD-L-test",
        "live_order_sent": True,
        "target_path_live_mode": "LIVE_LEARNING",
    }
    return {
        "event_uid": f"gateway-{trade_id}",
        "ts_utc": ts_utc,
        "event_type": "GATEWAY_ORDER_SENT",
        "lane_id": lane_id,
        "order_id": f"order-{trade_id}",
        "trade_id": trade_id,
        "pair": "EUR_USD",
        "side": "LONG",
        "units": 1000,
        "raw_json": {"target_path_receipt": receipt, "sent": True},
    }


def _original_side(close_side: str) -> str:
    side = str(close_side).upper()
    if side == "LONG":
        return "SHORT"
    if side == "SHORT":
        return "LONG"
    return side


def _units_for_side(side: str) -> int:
    return 1000 if str(side).upper() == "LONG" else -1000


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

    def test_profitable_unattributed_close_creates_user_alpha_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 1000,
                        "pl": None,
                        "event_type": "ORDER_FILLED",
                        "trade_id": "manual-eurusd-long",
                        "ts_utc": self._ts(4),
                    },
                    {
                        "pair": "EUR_USD",
                        "close_side": "SHORT",
                        "pl": 2300,
                        "event_type": "TRADE_CLOSED",
                        "trade_id": "manual-eurusd-long",
                        "ts_utc": self._ts(2),
                        "attributed": False,
                    },
                ],
            )

            report = compute_daily_review(path, now=self.now)

            self.assertEqual(report.bias_overrides, {})
            self.assertEqual(len(report.user_alpha_trades), 1)
            trade = report.user_alpha_trades[0]
            self.assertEqual(trade["edge_source"], "USER_ALPHA")
            self.assertEqual(trade["classification"], "OPERATOR_ALPHA")
            self.assertEqual(trade["pair"], "EUR_USD")
            self.assertEqual(trade["direction"], "LONG")
            self.assertFalse(trade["system_discovered"])
            self.assertEqual(trade["realized_pl_jpy"], 2300)
            self.assertEqual(trade["time_to_tp_seconds"], 7200)
            self.assertTrue(report.user_alpha_continuation["active"])
            self.assertEqual(
                report.user_alpha_continuation["five_pct_path_board_candidate"]["candidate_roles"],
                ["RELOAD", "SECOND_SHOT"],
            )
            self.assertIn("user-alpha: EUR_USD:LONG +2300JPY", report.narrative_summary)

            out_path = Path(tmp) / "trader_overrides.json"
            write_trader_overrides(report, out_path)
            data = json.loads(out_path.read_text())
            self.assertTrue(data["user_alpha_continuation"]["active"])
            self.assertEqual(data["_diagnostics"]["user_alpha_counts"], {"OPERATOR_ALPHA": 1})

    def test_gateway_attributed_winner_is_not_user_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            _make_db(
                path,
                [
                    {
                        "pair": "EUR_USD",
                        "close_side": "SHORT",
                        "pl": 2300,
                        "trade_id": "bot-eurusd-long",
                        "ts_utc": self._ts(2),
                    },
                ],
            )

            report = compute_daily_review(path, now=self.now)

            self.assertEqual(report.user_alpha_trades, [])
            self.assertFalse(report.user_alpha_continuation["active"])

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

    def test_structural_losing_direction_triggers_bias_when_recent_window_is_sparse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            trades = [
                {
                    "pair": "GBP_USD",
                    "close_side": "LONG",
                    "pl": -900,
                    "ts_utc": self._ts(30 + i),
                }
                for i in range(DAILY_REVIEW_STRUCTURAL_N_TRADES_FOR_BIAS)
            ]
            _make_db(path, trades)

            report = compute_daily_review(path, now=self.now, lookback_hours=24)

            self.assertIn("GBP_USD", report.bias_overrides)
            self.assertIn("SHORT", report.bias_overrides["GBP_USD"])
            self.assertLess(report.bias_overrides["GBP_USD"]["SHORT"], 0)
            self.assertIn("structural losing: GBP_USD:SHORT", report.narrative_summary)
            self.assertEqual(
                report.structural_pair_counts[("GBP_USD", "SHORT")],
                DAILY_REVIEW_STRUCTURAL_N_TRADES_FOR_BIAS,
            )

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

    def test_gateway_attributed_lane_blocks_when_close_lane_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.db"
            lane_id = "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:MARKET"
            trades = [
                {
                    "pair": "GBP_USD",
                    "close_side": "SHORT",
                    "pl": -1000,
                    "ts_utc": self._ts(20 - i),
                    "lane_id": None,
                    "entry_lane_id": lane_id,
                }
                for i in range(DAILY_REVIEW_N_LOSSES_FOR_BLOCK)
            ]
            _make_db(path, trades)

            report = compute_daily_review(path, now=self.now)

            self.assertIn(lane_id, report.blocked_lanes)
            self.assertEqual(report.lane_loss_counts[lane_id], DAILY_REVIEW_N_LOSSES_FOR_BLOCK)

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

    def test_live_target_path_review_classifies_good_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ledger.db"
            _make_target_path_review_db(
                db_path,
                [
                    _target_path_sent_row(ts_utc=self._ts(2), trade_id="tp-good"),
                    {
                        "event_uid": "close-tp-good",
                        "ts_utc": self._ts(1),
                        "event_type": "TRADE_CLOSED",
                        "lane_id": "target:EUR_USD:LONG:HERO",
                        "trade_id": "tp-good",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 1000,
                        "realized_pl_jpy": 1200.0,
                        "exit_reason": "TAKE_PROFIT",
                    },
                ],
            )

            report = compute_daily_review(db_path, now=self.now)
            data = report.to_dict()

            self.assertEqual(len(report.target_path_live_reviews), 1)
            review = report.target_path_live_reviews[0]
            self.assertEqual(review["classification"], "good execution")
            self.assertEqual(review["target_path_live_mode"], "LIVE_LEARNING")
            self.assertEqual(review["live_order_gateway_receipt_id"], "qrv1-EURUSD-L-test")
            self.assertEqual(data["_diagnostics"]["target_path_live_review_counts"], {"good execution": 1})
            self.assertIn("target-path live: good execution=1", report.narrative_summary)

    def test_live_target_path_review_keeps_unresolved_send_as_deployment_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ledger.db"
            _make_target_path_review_db(
                db_path,
                [_target_path_sent_row(ts_utc=self._ts(2), trade_id="tp-open")],
            )

            report = compute_daily_review(db_path, now=self.now)

            self.assertEqual(report.target_path_live_reviews[0]["classification"], "deployment failure")
            self.assertIsNone(report.target_path_live_reviews[0]["realized_pl_jpy"])

    def test_live_target_path_review_negative_without_exit_reason_is_vehicle_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ledger.db"
            _make_target_path_review_db(
                db_path,
                [
                    _target_path_sent_row(ts_utc=self._ts(2), trade_id="tp-loss"),
                    {
                        "event_uid": "close-tp-loss",
                        "ts_utc": self._ts(1),
                        "event_type": "TRADE_CLOSED",
                        "lane_id": "target:EUR_USD:LONG:HERO",
                        "trade_id": "tp-loss",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 1000,
                        "realized_pl_jpy": -400.0,
                    },
                ],
            )

            report = compute_daily_review(db_path, now=self.now)

            self.assertEqual(report.target_path_live_reviews[0]["classification"], "vehicle failure")

    def test_market_read_review_scores_prediction_accuracy_separately_from_trade_pl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ledger.db"
            _make_db(db_path, [])
            score_path = Path(tmp) / "market_read_predictions.jsonl"
            rows = [
                {
                    "generated_at_utc": self._ts(3),
                    "pair": "EUR_USD",
                    "direction": "LONG",
                    "action": "WAIT",
                    "verification_status": "REJECTED",
                    "verdict": "CORRECT",
                    "thirty_minute_verdict": "CORRECT",
                    "two_hour_verdict": "CORRECT",
                },
                {
                    "generated_at_utc": self._ts(2),
                    "pair": "GBP_USD",
                    "direction": "SHORT",
                    "action": "TRADE",
                    "verification_status": "ACCEPTED",
                    "verdict": "WRONG",
                    "thirty_minute_verdict": "WRONG",
                    "two_hour_verdict": "WRONG",
                },
            ]
            score_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            report = compute_daily_review(db_path, now=self.now, market_read_score_path=score_path)
            review = report.market_read_review

            self.assertEqual(report.bias_overrides, {})
            self.assertEqual(review["total_predictions"], 2)
            self.assertEqual(review["resolved_predictions"], 2)
            self.assertEqual(review["full_read_accuracy_pct"], 50.0)
            self.assertEqual(review["accuracy_30m_pct"], 50.0)
            self.assertEqual(review["accuracy_2h_pct"], 50.0)
            self.assertEqual(review["schema_v1_legacy_predictions"], 2)
            self.assertEqual(review["schema_v2_predictions"], 0)
            self.assertEqual(review["legacy_v1_full_target_accuracy_pct"], 50.0)
            self.assertEqual(review["blocked_but_correct_read_count"], 1)
            self.assertEqual(review["wrong_read_traded_count"], 1)
            self.assertIn(
                "market-read legacy-v1: 2/2 resolved excluded_geometry=0 "
                "full_target_accuracy=50.0 (not direction accuracy)",
                report.narrative_summary,
            )
            self.assertEqual(
                report.to_dict()["_diagnostics"]["market_read_review"]["verdict_counts"],
                {"CORRECT": 1, "WRONG": 1},
            )

    def test_market_read_review_excludes_legacy_inverted_geometry_from_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ledger.db"
            _make_db(db_path, [])
            score_path = Path(tmp) / "market_read_predictions.jsonl"
            rows = [
                {
                    "generated_at_utc": self._ts(3),
                    "pair": "EUR_USD",
                    "direction": "LONG",
                    "start_price": 1.1000,
                    "action": "WAIT",
                    "verification_status": "ACCEPTED",
                    "next_30m_prediction": {
                        "direction": "LONG",
                        "target_zone": "1.1100",
                        "invalidation": "1.0900",
                    },
                    "next_2h_prediction": {
                        "direction": "LONG",
                        "target_zone": "1.1150",
                        "invalidation": "1.0900",
                    },
                    "best_trade_if_forced": {
                        "direction": "LONG",
                        "entry": "1.1000",
                        "tp": "1.1100",
                        "sl": "1.0900",
                    },
                    "verdict": "CORRECT",
                    "thirty_minute_verdict": "CORRECT",
                    "two_hour_verdict": "CORRECT",
                },
                {
                    "generated_at_utc": self._ts(2),
                    "pair": "AUD_JPY",
                    "direction": "SHORT",
                    "start_price": 112.324,
                    "action": "REQUEST_EVIDENCE",
                    "verification_status": "ACCEPTED",
                    "next_30m_prediction": {
                        "direction": "SHORT",
                        "target_zone": "112.444",
                        "invalidation": "112.203",
                    },
                    "next_2h_prediction": {
                        "direction": "SHORT",
                        "target_zone": "112.444",
                        "invalidation": "112.203",
                    },
                    "best_trade_if_forced": {
                        "direction": "SHORT",
                        "entry": "112.284",
                        "tp": "112.444",
                        "sl": "112.203",
                    },
                    "verdict": "INVALIDATED_FIRST",
                    "thirty_minute_verdict": "INVALIDATED_FIRST",
                    "two_hour_verdict": "INVALIDATED_FIRST",
                },
            ]
            score_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            report = compute_daily_review(db_path, now=self.now, market_read_score_path=score_path)
            review = report.market_read_review

            self.assertEqual(review["total_predictions"], 2)
            self.assertEqual(review["score_eligible_predictions"], 1)
            self.assertEqual(review["geometry_ineligible_predictions"], 1)
            self.assertEqual(review["resolved_predictions"], 1)
            self.assertEqual(review["full_read_accuracy_pct"], 100.0)
            self.assertEqual(review["verdict_counts"], {"CORRECT": 1})
            self.assertIn(
                "market-read legacy-v1: 1/1 resolved excluded_geometry=1 "
                "full_target_accuracy=100.0 (not direction accuracy)",
                report.narrative_summary,
            )

    def test_market_read_review_excludes_straddling_directional_rails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ledger.db"
            _make_db(db_path, [])
            score_path = Path(tmp) / "market_read_predictions.jsonl"
            row = {
                "generated_at_utc": self._ts(2),
                "pair": "EUR_USD",
                "direction": "LONG",
                "start_price": 1.1000,
                "action": "WAIT",
                "verification_status": "ACCEPTED",
                "next_30m_prediction": {
                    "direction": "LONG",
                    "target_zone": "1.0990-1.1020",
                    "invalidation": "1.0980",
                },
                "next_2h_prediction": {
                    "direction": "LONG",
                    "target_zone": "1.0990-1.1040",
                    "invalidation": "1.0970",
                },
                "best_trade_if_forced": {
                    "direction": "LONG",
                    "entry": "1.1000",
                    "tp": "1.0990-1.1040",
                    "sl": "1.0970",
                },
                "verdict": "CORRECT",
                "thirty_minute_verdict": "TARGET_ONLY",
                "two_hour_verdict": "TARGET_ONLY",
            }
            score_path.write_text(json.dumps(row) + "\n")

            report = compute_daily_review(
                db_path,
                now=self.now,
                market_read_score_path=score_path,
            )

            review = report.market_read_review
            self.assertEqual(review["score_eligible_predictions"], 0)
            self.assertEqual(review["geometry_ineligible_predictions"], 1)
            self.assertEqual(review["resolved_predictions"], 0)
            self.assertIn(
                "market-read legacy-v1: 0/0 resolved excluded_geometry=1 "
                "full_target_accuracy=None (not direction accuracy)",
                report.narrative_summary,
            )

    def test_market_read_v2_reports_direction_target_and_full_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ledger.db"
            _make_db(db_path, [])
            score_path = Path(tmp) / "market_read_predictions.jsonl"

            def horizon(
                *,
                resolution: str = "RESOLVED_MID_CANDLE_DIAGNOSTIC",
                direction: str = "CORRECT",
                target: str = "NOT_TOUCHED",
                invalidation: str = "NOT_TOUCHED",
                first: str = "NEITHER_TOUCHED",
                full: str = "DIRECTION_CORRECT_TARGET_INCOMPLETE",
            ) -> dict:
                return {
                    "resolution_status": resolution,
                    "direction_status": direction,
                    "target_completion_status": target,
                    "invalidation_status": invalidation,
                    "first_touch_status": first,
                    "full_read_status": full,
                    "truth_source": "MID_CANDLE_DIAGNOSTIC",
                    "live_permission": False,
                }

            rows = [
                {
                    "schema_version": 2,
                    "prediction_id": "mr2:one",
                    "generated_at_utc": self._ts(3),
                    "pair": "EUR_USD",
                    "direction": "LONG",
                    "score_eligible": True,
                    "verdict": "FULL_READ_INCOMPLETE",
                    "horizon_results": {
                        "30m": horizon(),
                        "2h": horizon(
                            direction="WRONG",
                            target="TOUCHED",
                            first="TARGET_ONLY",
                            full="TARGET_ONLY",
                        ),
                    },
                    "reaction_chain": {
                        "first_subsequent_decision": {"status": "RESOLVED"},
                        "execution_attribution": {"status": "PARTIALLY_ATTRIBUTED"},
                        "realized_outcome": {"status": "RESOLVED", "realized_pl_jpy": 250.0},
                    },
                    "originating_decision_receipt_id": "gptd:one",
                    "direct_execution_attribution": {"status": "UNATTRIBUTED"},
                    "direct_realized_outcome": {"status": "UNRESOLVED"},
                },
                {
                    "schema_version": 2,
                    "prediction_id": "mr2:two",
                    "generated_at_utc": self._ts(2),
                    "pair": "GBP_USD",
                    "direction": "SHORT",
                    "score_eligible": True,
                    "verdict": "UNRESOLVED",
                    "horizon_results": {
                        "30m": horizon(
                            direction="WRONG",
                            invalidation="TOUCHED",
                            first="INVALIDATION_ONLY",
                            full="INVALIDATION_ONLY",
                        ),
                        "2h": horizon(
                            resolution="UNRESOLVED",
                            direction="UNRESOLVED",
                            target="UNRESOLVED",
                            invalidation="UNRESOLVED",
                            first="UNRESOLVED",
                            full="UNRESOLVED",
                        ),
                    },
                    "originating_decision_receipt_id": "gptd:two",
                    "direct_execution_attribution": {"status": "PARTIALLY_ATTRIBUTED"},
                    "direct_realized_outcome": {"status": "PARTIALLY_RESOLVED"},
                },
            ]
            score_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            report = compute_daily_review(db_path, now=self.now, market_read_score_path=score_path)
            review = report.market_read_review

            self.assertEqual(review["schema_v1_legacy_predictions"], 0)
            self.assertEqual(review["schema_v2_predictions"], 2)
            self.assertEqual(review["direction_accuracy_30m_pct"], 50.0)
            self.assertEqual(review["target_completion_30m_pct"], 0.0)
            self.assertEqual(review["invalidation_touch_30m_pct"], 50.0)
            self.assertEqual(review["full_read_completion_30m_pct"], 0.0)
            self.assertEqual(review["direction_accuracy_2h_pct"], 0.0)
            self.assertEqual(review["target_completion_2h_pct"], 100.0)
            self.assertEqual(review["full_read_completion_2h_pct"], 100.0)
            self.assertEqual(review["v2_resolved_horizons"], 3)
            self.assertEqual(review["v2_unresolved_horizons"], 1)
            self.assertEqual(
                review["direct_origin_counts"]["originating_decision_bound"],
                2,
            )
            self.assertEqual(
                review["direct_origin_counts"]["execution_partially_attributed"],
                1,
            )
            self.assertEqual(
                review["direct_origin_counts"]["realized_outcome_resolved"],
                0,
            )
            self.assertEqual(
                review["direct_origin_counts"]["realized_outcome_partially_resolved"],
                1,
            )
            self.assertEqual(
                review["reaction_chain_counts"]["first_subsequent_decision_resolved"],
                1,
            )
            self.assertEqual(review["reaction_chain_counts"]["execution_unattributed"], 1)
            self.assertEqual(review["reaction_chain_counts"]["realized_outcome_resolved"], 1)
            self.assertIsNone(review["full_read_accuracy_pct"])
            self.assertIn(
                "market-read-v2: 30m direction=50.0 target=0.0 full=0.0; "
                "2h direction=0.0 target=100.0 full=100.0 resolved_horizons=3 "
                "truth=MID_CANDLE_DIAGNOSTIC live_permission=false "
                "direct_origin=2/2 direct_execution_partially_attributed=1 "
                "direct_realized=0 direct_realized_partial=1; "
                "prior_reaction_decision=1/2 "
                "prior_reaction_execution_partially_attributed=1 "
                "prior_reaction_realized=1",
                report.narrative_summary,
            )


if __name__ == "__main__":
    unittest.main()
