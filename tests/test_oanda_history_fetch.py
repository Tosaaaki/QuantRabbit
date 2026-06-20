from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "oanda_history_fetch.py"
    spec = importlib.util.spec_from_file_location("oanda_history_fetch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hist = _load_module()


class OandaHistoryFetchTest(unittest.TestCase):
    def test_s5_chunks_stay_under_candle_cap(self) -> None:
        start = datetime(2026, 6, 1, tzinfo=timezone.utc)
        end = start + timedelta(hours=2)

        chunks = list(
            hist._iter_time_chunks(
                start,
                end,
                granularity="S5",
                max_candles_per_request=100,
            )
        )

        self.assertGreater(len(chunks), 1)
        for chunk_start, chunk_end in chunks:
            seconds = (chunk_end - chunk_start).total_seconds()
            self.assertLessEqual(seconds / 5, 99)

    def test_rows_preserve_bid_ask_and_skip_incomplete_by_default(self) -> None:
        payload = {
            "candles": [
                {
                    "time": "2026-06-01T00:00:00.000000000Z",
                    "complete": True,
                    "volume": 12,
                    "bid": {"o": "1.1000", "h": "1.1002", "l": "1.0998", "c": "1.1001"},
                    "ask": {"o": "1.1001", "h": "1.1003", "l": "1.0999", "c": "1.1002"},
                },
                {
                    "time": "2026-06-01T00:01:00.000000000Z",
                    "complete": False,
                    "volume": 3,
                    "bid": {"o": "1.1", "h": "1.1", "l": "1.1", "c": "1.1"},
                },
            ]
        }

        rows = list(
            hist._rows_from_payload(
                payload,
                pair="EUR_USD",
                granularity="M1",
                price="BA",
                include_incomplete=False,
            )
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pair"], "EUR_USD")
        self.assertEqual(rows[0]["bid"]["c"], 1.1001)
        self.assertEqual(rows[0]["ask"]["c"], 1.1002)

    def test_fetch_task_writes_jsonl_and_dedupes_boundary_rows(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def get_json(self, _path, _query):
                self.calls += 1
                ts = "2026-06-01T00:00:00.000000000Z"
                return {
                    "candles": [
                        {
                            "time": ts,
                            "complete": True,
                            "volume": self.calls,
                            "mid": {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.0"},
                        }
                    ]
                }

        start = datetime(2026, 6, 1, tzinfo=timezone.utc)
        end = start + timedelta(minutes=3)
        task = hist.FetchTask(pair="EUR_USD", granularity="M1", start=start, end=end, price="M")

        with tempfile.TemporaryDirectory() as tmp:
            summary = hist._fetch_task(
                FakeClient(),
                task,
                run_dir=Path(tmp),
                max_candles_per_request=2,
                sleep_seconds=0.0,
                retries=1,
                include_incomplete=False,
                dry_run=False,
            )

            self.assertGreater(summary["requests"], 1)
            self.assertEqual(summary["rows"], 1)
            rows = Path(summary["path"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["time"], "2026-06-01T00:00:00.000000000Z")


if __name__ == "__main__":
    unittest.main()
