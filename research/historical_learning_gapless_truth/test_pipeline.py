#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import lzma
from pathlib import Path
import struct
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("run_pipeline.py")
SPEC = importlib.util.spec_from_file_location("gapless_truth_pipeline", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

VERIFY_SPEC = importlib.util.spec_from_file_location("gapless_truth_verifier", MODULE_PATH.with_name("verify_pipeline.py"))
VERIFY = importlib.util.module_from_spec(VERIFY_SPEC)
sys.modules[VERIFY_SPEC.name] = VERIFY
VERIFY_SPEC.loader.exec_module(VERIFY)


class PipelineTest(unittest.TestCase):
    def test_required_scope_is_frozen(self) -> None:
        episodes = MODULE.read_episodes()
        self.assertEqual(251, len(episodes))
        self.assertEqual(146, sum(row["pair"] in MODULE.PAIRS for row in episodes))
        self.assertEqual(418, len(MODULE.required_hours(episodes)))

    def test_month_is_zero_based_in_official_datafeed_url(self) -> None:
        key = MODULE.HourKey("EUR_USD", datetime(2026, 7, 8, 12, tzinfo=timezone.utc))
        self.assertTrue(key.url.endswith("/EURUSD/2026/06/08/12h_ticks.bi5"))

    def test_decode_oracle_and_bid_ask_invariant(self) -> None:
        key = MODULE.HourKey("EUR_USD", datetime(2026, 7, 8, 12, tzinfo=timezone.utc))
        payload = b"".join([
            struct.pack(">3i2f", 57, 113983, 113981, 5.4, 3.6),
            struct.pack(">3i2f", 163, 113982, 113980, 1.8, 7.65),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.bi5"
            path.write_bytes(lzma.compress(payload))
            ticks, audit = MODULE.decode_hour(key, path)
        self.assertEqual(2, audit["rows"])
        self.assertAlmostEqual(1.13983, ticks[0].ask)
        self.assertAlmostEqual(1.13981, ticks[0].bid)
        self.assertLessEqual(ticks[0].bid, ticks[0].ask)
        self.assertLess(ticks[0].time, ticks[1].time)

    def test_crossed_quote_fails_closed(self) -> None:
        key = MODULE.HourKey("AUD_JPY", datetime(2026, 5, 6, 12, tzinfo=timezone.utc))
        payload = struct.pack(">3i2f", 1, 100000, 100001, 1.0, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.bi5"
            path.write_bytes(lzma.compress(payload))
            with self.assertRaisesRegex(RuntimeError, "crossed_quote"):
                MODULE.decode_hour(key, path)

    def test_non_monotonic_tick_fails_closed(self) -> None:
        key = MODULE.HourKey("EUR_JPY", datetime(2026, 6, 29, 12, tzinfo=timezone.utc))
        payload = b"".join([
            struct.pack(">3i2f", 10, 170001, 170000, 1.0, 1.0),
            struct.pack(">3i2f", 9, 170001, 170000, 1.0, 1.0),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.bi5"
            path.write_bytes(lzma.compress(payload))
            with self.assertRaisesRegex(RuntimeError, "non_monotonic_tick"):
                MODULE.decode_hour(key, path)

    def test_duplicate_records_are_counted_without_reordering(self) -> None:
        key = MODULE.HourKey("EUR_USD", datetime(2026, 7, 8, 12, tzinfo=timezone.utc))
        record = struct.pack(">3i2f", 57, 113983, 113981, 5.4, 3.6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.bi5"
            path.write_bytes(lzma.compress(record + record))
            ticks, audit = MODULE.decode_hour(key, path)
        self.assertEqual(2, len(ticks))
        self.assertEqual(1, audit["duplicate_timestamps"])
        self.assertEqual(1, audit["exact_duplicate_records"])

    def test_invalid_volume_fails_closed(self) -> None:
        key = MODULE.HourKey("EUR_USD", datetime(2026, 7, 8, 12, tzinfo=timezone.utc))
        payload = struct.pack(">3i2f", 57, 113983, 113981, float("nan"), 3.6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.bi5"
            path.write_bytes(lzma.compress(payload))
            with self.assertRaisesRegex(RuntimeError, "invalid_volume"):
                MODULE.decode_hour(key, path)

    def test_weekend_dst_boundary(self) -> None:
        self.assertTrue(MODULE.market_closed(datetime(2026, 7, 10, 21, tzinfo=timezone.utc)))
        self.assertTrue(MODULE.market_closed(datetime(2026, 7, 12, 20, tzinfo=timezone.utc)))
        self.assertFalse(MODULE.market_closed(datetime(2026, 7, 12, 21, tzinfo=timezone.utc)))
        self.assertTrue(MODULE.market_closed(datetime(2026, 1, 9, 22, tzinfo=timezone.utc)))
        self.assertFalse(MODULE.market_closed(datetime(2026, 1, 11, 22, tzinfo=timezone.utc)))

    def test_directory_size_tolerates_empty_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(0, MODULE.directory_bytes(Path(tmp)))

    def test_independent_metric_oracle_recomputes_selection(self) -> None:
        report = {
            "prediction_rows": [
                {"window_id": "W", "actual_net_jpy": 10.0, "price_action_selected": True},
                {"window_id": "W", "actual_net_jpy": -4.0, "price_action_selected": False},
            ],
            "windows": [{
                "id": "W", "status": "EVALUATED",
                "PRICE_ACTION_HGB": {
                    "trades_available": 2, "trades_selected": 1, "net_jpy": 10.0,
                    "baseline_net_jpy": 6.0, "incremental_net_jpy": 4.0,
                    "profit_factor": "Infinity", "max_drawdown_jpy": 0.0,
                    "paired_lcb_jpy": 0.0,
                },
            }],
        }
        result = VERIFY.metric_oracle(report)
        self.assertTrue(result["pass"])
        self.assertEqual(4.0, result["windows"][0]["oracle"]["incremental_net_jpy"])

    def test_persistent_feed_rejects_host_boundary_change(self) -> None:
        feed = MODULE.PersistentHistoricalFeed(MODULE.SOURCE_BASE)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "source host boundary changed"):
                feed.download("https://example.com/not-authorized.bi5", Path(tmp) / "x.bi5")

    def test_scheduled_market_close_does_not_call_downloader(self) -> None:
        key = MODULE.HourKey("EUR_USD", datetime(2026, 5, 10, 19, tzinfo=timezone.utc))
        calls = []
        row = MODULE.acquire_one(key, lambda *_: calls.append(True))
        self.assertEqual([], calls)
        self.assertFalse(row["complete"])
        self.assertTrue(row["market_closed"])
        self.assertEqual("MARKET_CLOSED", row["gap_reason"])
        self.assertEqual("SCHEDULED_MARKET_CLOSED_NO_FETCH", row["error"])


if __name__ == "__main__":
    unittest.main()
