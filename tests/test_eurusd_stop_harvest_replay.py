from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools import build_eurusd_stop_harvest_replay as builder


class EurusdStopHarvestReplayTest(unittest.TestCase):
    def test_short_stop_trigger_uses_bid_low_not_ask_low(self) -> None:
        candles = [
            _candle("2026-05-20T00:00:05.000000000Z", bid_low=1.1001, bid_high=1.1004, ask_low=1.0999),
            _candle("2026-05-20T00:00:10.000000000Z", bid_low=1.1000, bid_high=1.1003, ask_low=1.1002),
        ]

        touch = builder._first_short_stop_trigger(
            candles,
            start=builder._parse_time("2026-05-20T00:00:01Z"),
            end=builder._parse_time("2026-05-20T00:00:30Z"),
            trigger_price=1.1000,
        )

        self.assertIsNotNone(touch)
        assert touch is not None
        self.assertEqual(touch.candle.time_text, "2026-05-20T00:00:10.000000000Z")

    def test_build_payload_replays_stop_trigger_and_active_tp_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            history = root / "logs" / "replay" / "oanda_history" / "run" / "EUR_USD"
            data.mkdir(parents=True)
            history.mkdir(parents=True)
            _write_json(
                data / "eurusd_short_breakout_failure_vehicle_split_diagnosis.json",
                {
                    "vehicle_split": {
                        "stop_order_samples": [
                            {
                                "trade_id": "101",
                                "entry_order_type": "STOP_ORDER",
                                "realized_pl_jpy": 100.0,
                            }
                        ]
                    }
                },
            )
            db = data / "execution_ledger.db"
            _write_db(db)
            _write_history(
                history / "EUR_USD_S5_BA_20260520T000000Z_20260520T000100Z.jsonl"
            )

            payload = builder.build_payload(
                "2026-07-08T00:00:00Z",
                root=root,
                history_root=root / "logs" / "replay" / "oanda_history",
                ledger_db=db,
                vehicle_split_path=data / "eurusd_short_breakout_failure_vehicle_split_diagnosis.json",
            )

        self.assertEqual(
            payload["status"],
            "STOP_HARVEST_EXACT_S5_BIDASK_REPLAY_PASSED_STILL_BLOCKED",
        )
        self.assertFalse(payload["live_permission_allowed"])
        self.assertFalse(payload["scout_candidate_after_replay"])
        self.assertEqual(payload["s5_path_replay_summary"]["s5_wins"], 1)
        row = payload["replay_rows"][0]
        self.assertEqual(row["s5_first_trigger_touch_utc"], "2026-05-20T00:00:05.000000000Z")
        self.assertEqual(row["s5_first_tp_touch_order_id"], "103")
        self.assertEqual(row["s5_first_tp_touch_after_trigger_utc"], "2026-05-20T00:00:20.000000000Z")
        self.assertFalse(row["market_sample_mixed_in"])
        self.assertFalse(row["limit_sample_mixed_in"])
        blockers = {item["code"] for item in payload["remaining_blockers"]}
        self.assertIn("STOP_TRIGGER_INVALIDATION_NOT_SCOUT_READY", blockers)
        self.assertIn("NO_FRESH_GATEWAY_PERMISSION", blockers)


def _candle(
    time_text: str,
    *,
    bid_low: float,
    bid_high: float,
    ask_low: float,
    ask_high: float | None = None,
) -> builder.Candle:
    ask_high = ask_high if ask_high is not None else ask_low + 0.0002
    return builder.Candle(
        time=builder._parse_time(time_text),
        time_text=time_text,
        bid_open=bid_low,
        bid_high=bid_high,
        bid_low=bid_low,
        bid_close=bid_high,
        ask_open=ask_low,
        ask_high=ask_high,
        ask_low=ask_low,
        ask_close=ask_high,
    )


def _write_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            create table oanda_transactions (
                transaction_id text primary key,
                type text not null,
                time_utc text,
                batch_id text,
                request_id text,
                raw_json text not null,
                inserted_at_utc text not null
            )
            """
        )
        rows = [
            ("100", _stop_order("100", "2026-05-20T00:00:01.000000000Z", "1.10000")),
            ("101", _entry_fill("101", "100", "2026-05-20T00:00:08.000000000Z")),
            ("102", _tp_order("102", "101", "2026-05-20T00:00:08.000000000Z", "1.09900", "ON_FILL")),
            ("103", _tp_order("103", "101", "2026-05-20T00:00:20.000000000Z", "1.09950", "REPLACEMENT")),
            ("104", _exit_fill("104", "103", "101", "2026-05-20T00:00:26.000000000Z")),
        ]
        for transaction_id, payload in rows:
            conn.execute(
                """
                insert into oanda_transactions
                    (transaction_id, type, time_utc, batch_id, request_id, raw_json, inserted_at_utc)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    payload["type"],
                    payload["time"],
                    payload.get("batchID"),
                    None,
                    json.dumps(payload),
                    "2026-07-08T00:00:00Z",
                ),
            )


def _stop_order(transaction_id: str, time: str, price: str) -> dict[str, object]:
    return {
        "id": transaction_id,
        "type": "STOP_ORDER",
        "time": time,
        "instrument": "EUR_USD",
        "units": "-1000",
        "price": price,
        "reason": "CLIENT_ORDER",
        "triggerCondition": "DEFAULT",
    }


def _entry_fill(transaction_id: str, order_id: str, time: str) -> dict[str, object]:
    return {
        "id": transaction_id,
        "type": "ORDER_FILL",
        "time": time,
        "instrument": "EUR_USD",
        "units": "-1000",
        "price": "1.10000",
        "pl": "0.0000",
        "quotePL": "0",
        "halfSpreadCost": "10.0000",
        "reason": "STOP_ORDER",
        "orderID": order_id,
        "fullPrice": _full_price("1.10000", "1.10020"),
    }


def _tp_order(transaction_id: str, trade_id: str, time: str, price: str, reason: str) -> dict[str, object]:
    return {
        "id": transaction_id,
        "type": "TAKE_PROFIT_ORDER",
        "time": time,
        "tradeID": trade_id,
        "price": price,
        "reason": reason,
        "triggerCondition": "DEFAULT",
    }


def _exit_fill(transaction_id: str, order_id: str, trade_id: str, time: str) -> dict[str, object]:
    return {
        "id": transaction_id,
        "type": "ORDER_FILL",
        "time": time,
        "instrument": "EUR_USD",
        "units": "1000",
        "price": "1.09950",
        "pl": "100.0000",
        "quotePL": "1.0",
        "halfSpreadCost": "10.0000",
        "reason": "TAKE_PROFIT_ORDER",
        "orderID": order_id,
        "fullPrice": _full_price("1.09930", "1.09950"),
        "tradesClosed": [{"tradeID": trade_id, "realizedPL": "100.0000", "units": "1000"}],
    }


def _full_price(bid: str, ask: str) -> dict[str, object]:
    return {
        "bids": [{"price": bid, "liquidity": "1000000"}],
        "asks": [{"price": ask, "liquidity": "1000000"}],
    }


def _write_history(path: Path) -> None:
    rows = [
        _history_row("2026-05-20T00:00:05.000000000Z", bid_low=1.0999, bid_high=1.1001, ask_low=1.1001, ask_high=1.1003),
        _history_row("2026-05-20T00:00:10.000000000Z", bid_low=1.0998, bid_high=1.1000, ask_low=1.0994, ask_high=1.1002),
        _history_row("2026-05-20T00:00:15.000000000Z", bid_low=1.0998, bid_high=1.1000, ask_low=1.0994, ask_high=1.1002),
        _history_row("2026-05-20T00:00:20.000000000Z", bid_low=1.0996, bid_high=1.0999, ask_low=1.0994, ask_high=1.1001),
        _history_row("2026-05-20T00:00:25.000000000Z", bid_low=1.0995, bid_high=1.0998, ask_low=1.0994, ask_high=1.0999),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _history_row(time: str, *, bid_low: float, bid_high: float, ask_low: float, ask_high: float) -> dict[str, object]:
    return {
        "pair": "EUR_USD",
        "granularity": "S5",
        "price": "BA",
        "time": time,
        "complete": True,
        "volume": 1,
        "bid": {"o": bid_high, "h": bid_high, "l": bid_low, "c": bid_high},
        "ask": {"o": ask_high, "h": ask_high, "l": ask_low, "c": ask_high},
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
