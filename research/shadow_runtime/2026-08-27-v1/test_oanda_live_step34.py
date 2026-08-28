from __future__ import annotations

import ast
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from oanda_live_feed import OandaLiveRecorder
from oanda_live_step34 import (
    CANARY_ARM,
    REST_HOST,
    STEP_LEDGERS,
    Step34Runtime,
    candle_content_hash,
    fetch_completed_m5,
)

ROOT = Path(__file__).resolve().parent


def quote(symbol: str, source: datetime, arrival: datetime, bid: float, ask: float) -> dict:
    return {
        "event_id": f"{symbol}-{source.isoformat()}",
        "instrument": symbol,
        "event_time_utc": source.isoformat().replace("+00:00", "Z"),
        "arrival_time_utc": arrival.isoformat().replace("+00:00", "Z"),
        "bid": bid,
        "ask": ask,
    }


class FakeResponse:
    status = 200

    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class FakeConnection:
    instances = []

    def __init__(self, host, **kwargs):
        self.host = host
        self.request_args = None
        self.closed = False
        self.__class__.instances.append(self)

    def request(self, method, path, headers):
        self.request_args = (method, path, headers)

    def getresponse(self):
        return FakeResponse({
            "instrument": "EUR_USD",
            "granularity": "M5",
            "candles": [
                {"time": "2026-08-28T00:00:00.000000000Z", "complete": True, "bid": {"o": "1.1"}, "ask": {"o": "1.2"}},
                {"time": "2026-08-28T00:05:00.000000000Z", "complete": False, "bid": {"o": "1.1"}, "ask": {"o": "1.2"}},
            ],
        })

    def close(self):
        self.closed = True


class Step34Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = Step34Runtime(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_historical_get_records_only_completed_and_excludes_forward_pnl(self):
        recorder = OandaLiveRecorder(self.root / "recorder")
        count = fetch_completed_m5(
            "not-a-real-account", "not-a-real-token", "EUR_USD", 2, recorder, self.runtime, FakeConnection
        )
        self.assertEqual(count, 1)
        row = self.runtime.ledgers["historical_warmup_m5"].rows[0]["payload"]
        self.assertTrue(row["complete"])
        self.assertTrue(row["excluded_from_forward_pnl"])
        self.assertEqual(row["content_sha256"], candle_content_hash({
            "time": "2026-08-28T00:00:00.000000000Z", "complete": True,
            "bid": {"o": "1.1"}, "ask": {"o": "1.2"},
        }))
        method, path, _ = FakeConnection.instances[-1].request_args
        self.assertEqual(method, "GET")
        self.assertIn("/candles?", path)
        self.assertIn("granularity=M5", path)

    def test_live_m5_closes_only_on_later_bucket_and_freezes_chronology(self):
        base = datetime(2026, 8, 28, 0, 0, 1, tzinfo=timezone.utc)
        self.runtime.on_stream_event(quote("EUR_USD", base, base + timedelta(milliseconds=10), 1.1, 1.2))
        self.assertEqual(self.runtime.status()["forward_completed_m5"], 0)
        later = datetime(2026, 8, 28, 0, 5, 1, tzinfo=timezone.utc)
        self.runtime.on_stream_event(quote("EUR_USD", later, later + timedelta(milliseconds=10), 1.11, 1.21))
        row = self.runtime.ledgers["forward_completed_m5"].rows[0]["payload"]
        self.assertTrue(row["source_arrival_chronology_frozen"])
        self.assertTrue(row["completed_from_live_price_events"])
        self.assertFalse(row["historical_warmup_used_for_pnl"])

    def test_accounting_only_r5_stays_zero_and_canary_uses_first_post_decision_bbo(self):
        base = datetime(2026, 8, 28, 0, 0, 1, tzinfo=timezone.utc)
        for symbol, offset in (("EUR_USD", 0), ("USD_JPY", 1)):
            stamp = base + timedelta(seconds=offset)
            self.runtime.on_stream_event(quote(symbol, stamp, stamp + timedelta(milliseconds=10), 1.1, 1.2))
        boundary = datetime(2026, 8, 28, 0, 5, 1, tzinfo=timezone.utc)
        self.runtime.on_stream_event(quote("EUR_USD", boundary, boundary + timedelta(milliseconds=10), 1.11, 1.21))
        self.runtime.on_stream_event(quote("USD_JPY", boundary + timedelta(seconds=1), boundary + timedelta(seconds=1, milliseconds=10), 147.0, 147.02))
        status = self.runtime.status()
        self.assertEqual(status["natural_r5_proposals"], 0)
        self.assertEqual(status["canary_proposals"], 1)
        self.assertEqual(status["virtual_fills"], 0)
        fill_event = quote("EUR_USD", boundary + timedelta(seconds=2), boundary + timedelta(seconds=2, milliseconds=10), 1.12, 1.13)
        self.runtime.on_stream_event(fill_event)
        status = self.runtime.status()
        self.assertEqual(status["decisions"], 1)
        self.assertEqual(status["virtual_fills"], 2)
        self.assertEqual(status["inventory_records"], 2)
        self.assertEqual(status["pnl_records"], 2)
        for ledger in ("canary_non_evidence", "decisions", "virtual_fills", "pnl"):
            for row in self.runtime.ledgers[ledger].rows:
                self.assertFalse(row["payload"]["profit_evidence"])
        fills = self.runtime.ledgers["virtual_fills"].rows
        self.assertTrue(all(row["payload"]["first_post_decision_bbo_event_id"] == fill_event["event_id"] for row in fills))
        self.assertTrue(all(row["payload"]["canary_arm"] == CANARY_ARM for row in fills))

    def test_restart_is_idempotent_after_canary_completion(self):
        for ledger in STEP_LEDGERS:
            payload = {"ledger": ledger}
            if ledger == "forward_completed_m5":
                payload["instrument"] = "EUR_USD"
            self.runtime.append(ledger, payload, f"fixture::{ledger}")
        restarted = Step34Runtime(self.root)
        for ledger in STEP_LEDGERS:
            self.assertEqual(len(restarted.ledgers[ledger].rows), 1)

    def test_source_surface_is_get_only_and_has_no_order_endpoints(self):
        source = (ROOT / "oanda_live_step34.py").read_text()
        tree = ast.parse(source)
        methods = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value in {"GET", "POST", "PUT", "PATCH", "DELETE"}
        }
        self.assertEqual(methods, {"GET"})
        self.assertEqual(REST_HOST, "https://api-fxtrade.oanda.com")
        self.assertIn('REST_NETLOC = "api-fxtrade.oanda.com"', source)
        lowered = source.lower()
        for banned in ("/orders", "/trades", "/positions", "broker adapter"):
            self.assertNotIn(banned, lowered)


if __name__ == "__main__":
    unittest.main()
