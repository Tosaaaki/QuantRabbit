from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.technical_forecast_forward_truth import fetch_frozen_s5_truth


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "resolve_technical_forecast_forward_outcomes.py"


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get_json(self, path: str, query: dict[str, str] | None = None):
        assert query is not None
        self.calls.append((path, dict(query)))
        timestamp = query["from"]
        return {
            "instrument": path.split("/")[-2],
            "granularity": "S5",
            "candles": [
                {
                    "time": timestamp,
                    "complete": True,
                    "bid": {"o": "1.0000", "h": "1.0002", "l": "0.9998", "c": "1.0001"},
                    "ask": {"o": "1.0001", "h": "1.0003", "l": "0.9999", "c": "1.0002"},
                }
            ]
        }


class TechnicalForecastForwardTruthTest(unittest.TestCase):
    def test_fixed_day_is_fetched_in_bounded_s5_chunks(self) -> None:
        client = _FakeClient()
        candles, hashes = fetch_frozen_s5_truth(
            client,
            pair="EUR_USD",
            time_from=datetime(2026, 7, 16, tzinfo=timezone.utc),
            time_to=datetime(2026, 7, 17, tzinfo=timezone.utc),
            chunk_candle_limit=4500,
        )

        self.assertEqual(len(client.calls), 4)
        self.assertEqual(len(candles), 4)
        self.assertEqual(len(hashes), 4)
        self.assertTrue(all(path == "/v3/instruments/EUR_USD/candles" for path, _ in client.calls))
        self.assertTrue(all(query["price"] == "BA" for _, query in client.calls))

    def test_non_grid_interval_is_aligned_inward_to_complete_s5_candles(self) -> None:
        client = _FakeClient()
        start = datetime(2026, 7, 16, 12, 0, 1, 250000, tzinfo=timezone.utc)
        end = start + timedelta(seconds=16.5)

        candles, hashes = fetch_frozen_s5_truth(
            client,
            pair="EUR_USD",
            time_from=start,
            time_to=end,
            chunk_candle_limit=4500,
        )

        self.assertEqual(len(candles), 1)
        self.assertEqual(len(hashes), 1)
        query = client.calls[0][1]
        self.assertEqual(query["from"], "2026-07-16T12:00:05.000000Z")
        self.assertEqual(query["to"], "2026-07-16T12:00:15.000000Z")
        self.assertEqual(
            candles[0].timestamp_utc,
            datetime(2026, 7, 16, 12, 0, 5, tzinfo=timezone.utc),
        )

    def test_no_shadow_ledger_never_initializes_oanda(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "scorecard.json"
            env = os.environ.copy()
            env.pop("QR_OANDA_TOKEN", None)
            env.pop("QR_OANDA_ACCOUNT_ID", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--shadow-ledger",
                    str(root / "missing.jsonl"),
                    "--outcome-ledger",
                    str(root / "outcomes.jsonl"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "NO_SHADOW_LEDGER")
        self.assertFalse(result["broker_read"])
        self.assertFalse(result["broker_mutation"])
        self.assertTrue(result["output_unchanged"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
