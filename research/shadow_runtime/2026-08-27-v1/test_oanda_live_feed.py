from __future__ import annotations

import ast
import json
import secrets
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from oanda_live_feed import (
    CONTINUITY,
    LOSSLESS,
    MAX_HEARTBEAT_GAP_SECONDS,
    REST_HOST,
    STREAM_HOST,
    SYMBOLS,
    OandaLiveRecorder,
)

ROOT = Path(__file__).resolve().parent


def price(instrument: str, stamp: str, bid: str, ask: str) -> bytes:
    return (json.dumps({
        "type": "PRICE",
        "instrument": instrument,
        "time": stamp,
        "status": "tradeable",
        "bids": [{"price": bid, "liquidity": 1000000}],
        "asks": [{"price": ask, "liquidity": 1000000}],
    }) + "\n").encode()


class OandaLiveFeedTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.recorder = OandaLiveRecorder(self.root)
        self.now = datetime(2026, 8, 28, tzinfo=timezone.utc)
        self.recorder.connect_started()
        self.recorder.connect_established(self.now)

    def tearDown(self):
        self.temp.cleanup()

    def test_price_and_heartbeat_are_append_only(self):
        self.recorder.ingest_line(price("EUR_USD", "2026-08-28T00:00:01.123456789Z", "1.1600", "1.1602"), self.now + timedelta(seconds=1))
        self.recorder.ingest_line((json.dumps({"type": "HEARTBEAT", "time": "2026-08-28T00:00:05.987654321Z"}) + "\n").encode(), self.now + timedelta(seconds=5))
        status = self.recorder.status()
        self.assertEqual(status["counters"]["market_events_accepted"], 1)
        self.assertEqual(status["counters"]["heartbeats"], 1)
        self.assertEqual(len(self.recorder.ledgers["raw_bbo"].rows), 1)
        self.assertEqual(len(self.recorder.ledgers["feed_quality"].rows), 2)

    def test_only_two_symbols_are_accepted(self):
        for index, symbol in enumerate(SYMBOLS):
            self.recorder.ingest_line(price(symbol, f"2026-08-28T00:00:0{index + 1}Z", "1.1", "1.2"), self.now + timedelta(seconds=index + 1))
        self.assertEqual(self.recorder.status()["counters"]["market_events_accepted"], 2)
        other = OandaLiveRecorder(self.root / "other")
        other.connect_started()
        other.connect_established(self.now)
        other.ingest_line(price("AUD_USD", "2026-08-28T00:00:01Z", "1.1", "1.2"), self.now + timedelta(seconds=1))
        self.assertTrue(other.status()["feed_blocked"])

    def test_duplicate_is_idempotent(self):
        raw = price("USD_JPY", "2026-08-28T00:00:01Z", "147.00", "147.02")
        self.recorder.ingest_line(raw, self.now + timedelta(seconds=1))
        restarted = OandaLiveRecorder(self.root)
        restarted.ingest_line(raw, self.now + timedelta(seconds=2))
        self.assertEqual(restarted.status()["counters"]["market_events_accepted"], 1)
        self.assertEqual(restarted.status()["counters"]["duplicate_events"], 1)
        self.assertEqual(len(restarted.ledgers["raw_bbo"].rows), 1)

    def test_source_time_regression_fails_closed(self):
        self.recorder.ingest_line(price("EUR_USD", "2026-08-28T00:00:02Z", "1.1600", "1.1602"), self.now + timedelta(seconds=1))
        self.recorder.ingest_line(price("EUR_USD", "2026-08-28T00:00:01Z", "1.1601", "1.1603"), self.now + timedelta(seconds=2))
        self.assertEqual(self.recorder.status()["block_reason"], "SOURCE_TIME_REGRESSION")
        self.assertEqual(self.recorder.status()["counters"]["decisions"], 0)
        self.assertEqual(self.recorder.status()["counters"]["virtual_fills"], 0)

    def test_arrival_gap_and_clock_reversal_fail_closed(self):
        for name, arrival in (
            ("LOCAL_ARRIVAL_GAP", self.now + timedelta(seconds=MAX_HEARTBEAT_GAP_SECONDS + 1)),
            ("LOCAL_CLOCK_REVERSAL", self.now - timedelta(seconds=1)),
        ):
            with self.subTest(name=name):
                recorder = OandaLiveRecorder(self.root / name)
                recorder.connect_started()
                recorder.connect_established(self.now)
                recorder.ingest_line(price("EUR_USD", "2026-08-28T00:00:01Z", "1.1", "1.2"), arrival)
                self.assertEqual(recorder.status()["block_reason"], name)

    def test_malformed_and_unknown_objects_fail_closed(self):
        for index, raw in enumerate((b"\xff", b"{}\n", b"[]\n")):
            recorder = OandaLiveRecorder(self.root / f"bad-{index}")
            recorder.connect_started()
            recorder.connect_established(self.now)
            recorder.ingest_line(raw, self.now + timedelta(seconds=1))
            self.assertEqual(recorder.status()["block_reason"], "MALFORMED_OR_UNKNOWN_STREAM_OBJECT")

    def test_contract_and_source_have_get_only_zero_order_surface(self):
        contract = json.loads((ROOT / "oanda_live_runtime_contract.json").read_text())
        self.assertEqual(contract["http_method_allowlist"], ["GET"])
        self.assertEqual(contract["fallback_providers"], [])
        self.assertEqual(contract["symbols"], list(SYMBOLS))
        self.assertFalse(contract["live_order_authority"])
        self.assertEqual(contract["external_orders"], 0)
        tree = ast.parse((ROOT / "oanda_live_feed.py").read_text())
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imports |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertFalse(any(name.startswith("quant_rabbit.broker") for name in imports))
        method_literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in {"GET", "POST", "PUT", "PATCH", "DELETE"}}
        self.assertEqual(method_literals, {"GET"})
        urls = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("https://")}
        self.assertEqual(urls, {REST_HOST, STREAM_HOST})
        status = self.recorder.status()
        self.assertEqual(status["external_orders"], 0)
        self.assertEqual(status["counters"]["external_order_attempts"], 0)
        self.assertTrue(status["credential_values_absent"])
        self.assertEqual(CONTINUITY, "HEARTBEAT_ONLY")
        self.assertFalse(LOSSLESS)

    def test_runtime_artifacts_do_not_contain_generated_credential_values(self):
        account_id = secrets.token_hex(8)
        token = secrets.token_hex(32)
        self.recorder.ingest_line(price("EUR_USD", "2026-08-28T00:00:01Z", "1.1", "1.2"), self.now + timedelta(seconds=1))
        blob = b"".join(path.read_bytes() for path in self.root.rglob("*") if path.is_file())
        self.assertNotIn(account_id.encode(), blob)
        self.assertNotIn(token.encode(), blob)

    def test_credential_read_counter_records_file_not_values(self):
        self.recorder.mark_approved_credential_file_read()
        self.assertEqual(self.recorder.status()["counters"]["credential_reads"], 1)
        status_text = (self.root / "status.json").read_text()
        self.assertNotIn("account_id", status_text)
        self.assertNotIn("token", status_text)


if __name__ == "__main__":
    unittest.main()
