from __future__ import annotations

import ast
import contextlib
import io
import json
import secrets
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import oanda_live_feed as feed_module
from oanda_live_feed import (
    CONTINUITY,
    LOSSLESS,
    MAX_HEARTBEAT_GAP_SECONDS,
    REST_HOST,
    STREAM_HOST,
    SYMBOLS,
    OandaLiveRecorder,
)
from shadow_runtime import IntegrityError, utc_text

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
        self.feed_attestation = "a" * 64
        self.recorder.connect_started(self.feed_attestation)
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
        raw = self.recorder.ledgers["raw_bbo"].rows[0]["payload"]
        self.assertEqual(raw["segment_id"], "segment-00000001")
        self.assertEqual(raw["segment_started_at_utc"], utc_text(self.now))
        self.assertEqual(raw["feed_service_attestation_hash"], self.feed_attestation)
        self.assertEqual(raw["feed_provenance_status"], "ATTESTED")
        connection = next(
            row["payload"] for row in self.recorder.ledgers["control"].rows
            if row["payload"].get("event") == "LIVE_PRICING_CONNECTED"
        )
        self.assertEqual(connection["segment_id"], raw["segment_id"])
        self.assertEqual(connection["segment_started_at_utc"], raw["segment_started_at_utc"])
        self.assertEqual(connection["feed_service_attestation_hash"], self.feed_attestation)
        heartbeat = next(
            row["payload"] for row in self.recorder.ledgers["feed_quality"].rows
            if row["payload"].get("type") == "HEARTBEAT"
        )
        self.assertEqual(heartbeat["feed_service_attestation_hash"], self.feed_attestation)

    def test_missing_or_invalid_feed_service_attestation_fails_closed(self):
        for index, value in enumerate((None, "not-a-sha256", "A" * 64)):
            with self.subTest(value=value):
                recorder = OandaLiveRecorder(self.root / f"attestation-{index}")
                with self.assertRaises(IntegrityError):
                    recorder.connect_started(value)  # type: ignore[arg-type]
                self.assertEqual(len(recorder.ledgers["raw_bbo"].rows), 0)
                self.assertEqual(len(recorder.ledgers["control"].rows), 0)

    def test_run_live_requires_attestation_before_any_network_or_ledger_write(self):
        recorder = OandaLiveRecorder(self.root / "run-live-no-attestation")
        with patch.object(feed_module.http.client, "HTTPSConnection") as connection:
            with self.assertRaisesRegex(IntegrityError, "FEED_SERVICE_ATTESTATION_REQUIRED"):
                recorder.run_live("unused-account", "unused-token", 0.01, runtime_hash=None)
        self.assertFalse(connection.called)
        status = recorder.status()
        self.assertEqual(status["counters"]["network_attempts"], 0)
        self.assertEqual(status["counters"]["credential_reads"], 0)
        self.assertEqual(len(recorder.ledgers["raw_bbo"].rows), 0)
        self.assertEqual(len(recorder.ledgers["control"].rows), 0)

    def test_direct_main_is_disabled_before_credentials_network_or_attested_rows(self):
        direct_root = self.root / "direct-main-disabled"
        stderr = io.StringIO()
        with (
            patch.object(feed_module, "load_approved_live_credentials") as credentials,
            patch.object(feed_module.http.client, "HTTPSConnection") as connection,
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = feed_module.main([
                "--runtime-root", str(direct_root),
                "--seconds", "0.01",
            ])
        self.assertEqual(exit_code, 3)
        self.assertFalse(credentials.called)
        self.assertFalse(connection.called)
        self.assertIn("FEED_SERVICE_ATTESTATION_REQUIRED", stderr.getvalue())
        self.assertIn("oanda_launchd_runtime.py feed", stderr.getvalue())
        self.assertFalse(direct_root.exists())

    def test_reconnect_assigns_new_segment_identity_to_raw_events(self):
        first = self.recorder.ingest_line(
            price("EUR_USD", "2026-08-28T00:00:01Z", "1.1600", "1.1602"),
            self.now + timedelta(seconds=1),
        )
        self.recorder.invalidate("STREAM_EOF", self.now + timedelta(seconds=2))
        self.recorder.connect_started(self.feed_attestation)
        reconnect_at = self.now + timedelta(seconds=3)
        self.recorder.connect_established(reconnect_at)
        second = self.recorder.ingest_line(
            price("EUR_USD", "2026-08-28T00:00:04Z", "1.1601", "1.1603"),
            self.now + timedelta(seconds=4),
        )
        self.assertNotEqual(first["segment_id"], second["segment_id"])
        self.assertEqual(second["segment_id"], "segment-00000002")
        self.assertEqual(second["segment_started_at_utc"], utc_text(reconnect_at))
        rows = [row["payload"] for row in self.recorder.ledgers["raw_bbo"].rows]
        self.assertEqual([row["segment_id"] for row in rows], ["segment-00000001", "segment-00000002"])

    def test_identical_market_event_replayed_after_reconnect_is_idempotent(self):
        raw = price("USD_JPY", "2026-08-28T00:00:01Z", "147.00", "147.02")
        self.recorder.ingest_line(raw, self.now + timedelta(seconds=1))
        self.recorder.invalidate("STREAM_EOF", self.now + timedelta(seconds=2))
        self.recorder.connect_started(self.feed_attestation)
        self.recorder.connect_established(self.now + timedelta(seconds=3))
        self.recorder.ingest_line(raw, self.now + timedelta(seconds=4))
        self.assertFalse(self.recorder.status()["feed_blocked"])
        self.assertEqual(self.recorder.status()["counters"]["duplicate_events"], 1)
        self.assertEqual(len(self.recorder.ledgers["raw_bbo"].rows), 1)

    def test_repeated_stream_eof_receipts_are_scoped_to_segments(self):
        self.recorder.invalidate("STREAM_EOF", self.now + timedelta(seconds=1))
        self.recorder.connect_started(self.feed_attestation)
        self.recorder.connect_established(self.now + timedelta(seconds=2))
        self.recorder.invalidate("STREAM_EOF", self.now + timedelta(seconds=3))
        invalidations = [
            row for row in self.recorder.ledgers["control"].rows
            if row["payload"].get("event") == "FEED_INVALID"
        ]
        self.assertEqual(len(invalidations), 2)
        self.assertEqual(
            {row["payload"]["segment_id"] for row in invalidations},
            {"segment-00000001", "segment-00000002"},
        )

    def test_heartbeat_replay_is_idempotent_within_segment_and_distinct_across_segments(self):
        raw = (json.dumps({"type": "HEARTBEAT", "time": "2026-08-28T00:00:05Z"}) + "\n").encode()
        self.recorder.ingest_line(raw, self.now + timedelta(seconds=1))
        replay = self.recorder.ingest_line(raw, self.now + timedelta(seconds=2))
        self.assertTrue(replay["duplicate"])
        self.assertEqual(self.recorder.status()["counters"]["heartbeats"], 1)
        self.assertEqual(self.recorder.status()["counters"]["duplicate_heartbeats"], 1)
        self.recorder.invalidate("STREAM_EOF", self.now + timedelta(seconds=3))
        self.recorder.connect_started(self.feed_attestation)
        self.recorder.connect_established(self.now + timedelta(seconds=4))
        self.assertFalse(self.recorder.status()["heartbeat_current"])
        self.recorder.ingest_line(raw, self.now + timedelta(seconds=5))
        self.assertTrue(self.recorder.status()["heartbeat_current"])
        heartbeats = [
            row for row in self.recorder.ledgers["feed_quality"].rows
            if row["payload"].get("type") == "HEARTBEAT"
        ]
        self.assertEqual(len(heartbeats), 2)
        self.assertEqual(
            {row["payload"]["segment_id"] for row in heartbeats},
            {"segment-00000001", "segment-00000002"},
        )

    def test_only_two_symbols_are_accepted(self):
        for index, symbol in enumerate(SYMBOLS):
            self.recorder.ingest_line(price(symbol, f"2026-08-28T00:00:0{index + 1}Z", "1.1", "1.2"), self.now + timedelta(seconds=index + 1))
        self.assertEqual(self.recorder.status()["counters"]["market_events_accepted"], 2)
        other = OandaLiveRecorder(self.root / "other")
        other.connect_started(self.feed_attestation)
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
        self.assertEqual(restarted.state["seen_event_ids"], {})

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
                recorder.connect_started(self.feed_attestation)
                recorder.connect_established(self.now)
                recorder.ingest_line(price("EUR_USD", "2026-08-28T00:00:01Z", "1.1", "1.2"), arrival)
                self.assertEqual(recorder.status()["block_reason"], name)

    def test_malformed_and_unknown_objects_fail_closed(self):
        for index, raw in enumerate((b"\xff", b"{}\n", b"[]\n")):
            recorder = OandaLiveRecorder(self.root / f"bad-{index}")
            recorder.connect_started(self.feed_attestation)
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
