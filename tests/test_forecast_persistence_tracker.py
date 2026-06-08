from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from quant_rabbit.strategy.forecast_persistence_tracker import (
    assess_position,
    record_forecast,
)


def _forecast(pair: str, direction: str, confidence: float = 0.8) -> SimpleNamespace:
    return SimpleNamespace(
        pair=pair,
        direction=direction,
        confidence=confidence,
        invalidation_price=None,
        target_price=None,
        horizon_min=60,
    )


class ForecastPersistenceTrackerTest(unittest.TestCase):
    def test_record_forecast_compacts_historical_cycle_pair_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "forecast_history.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "timestamp_utc": "2026-06-01T00:00:00Z",
                            "cycle_id": "cycle-old",
                            "pair": "EUR_USD",
                            "direction": "DOWN",
                            "confidence": 0.4,
                        },
                        {
                            "timestamp_utc": "2026-06-01T00:01:00Z",
                            "cycle_id": "cycle-old",
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "confidence": 0.8,
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            recorded = record_forecast(
                _forecast("GBP_USD", "DOWN"),
                data_root=root,
                cycle_id="cycle-new",
            )

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            old_rows = [
                row
                for row in rows
                if row.get("cycle_id") == "cycle-old" and row.get("pair") == "EUR_USD"
            ]
            self.assertTrue(recorded)
            self.assertEqual(len(old_rows), 1)
            self.assertEqual(old_rows[0]["direction"], "UP")
            self.assertEqual(rows[-1]["pair"], "GBP_USD")

    def test_record_forecast_compacts_existing_key_before_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "forecast_history.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "timestamp_utc": "2026-06-01T00:00:00Z",
                            "cycle_id": "cycle-1",
                            "pair": "EUR_USD",
                            "direction": "DOWN",
                            "confidence": 0.4,
                        },
                        {
                            "timestamp_utc": "2026-06-01T00:01:00Z",
                            "cycle_id": "cycle-1",
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "confidence": 0.8,
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            recorded = record_forecast(
                _forecast("EUR_USD", "DOWN"),
                data_root=root,
                cycle_id="cycle-1",
            )

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertFalse(recorded)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["direction"], "UP")

    def test_duplicate_pair_forecasts_in_one_cycle_count_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for _ in range(3):
                record_forecast(
                    _forecast("EUR_USD", "DOWN"),
                    data_root=root,
                    cycle_id="cycle-1",
                )

            verdict = assess_position(
                trade_id="t1",
                pair="EUR_USD",
                side="LONG",
                data_root=root,
            )

            self.assertEqual(verdict.verdict, "HOLD")
            self.assertEqual(verdict.last_n_directions, ("DOWN",))
            rows = (root / "forecast_history.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)

    def test_duplicate_pair_forecasts_race_counts_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            thread_count = 6
            barrier = threading.Barrier(thread_count)

            def stale_precheck(*_args: object, **_kwargs: object) -> bool:
                barrier.wait(timeout=5)
                return False

            results: list[bool] = []
            errors: list[BaseException] = []

            def worker() -> None:
                try:
                    results.append(
                        record_forecast(
                            _forecast("EUR_USD", "DOWN"),
                            data_root=root,
                            cycle_id="cycle-race",
                        )
                    )
                except BaseException as exc:
                    errors.append(exc)

            with mock.patch(
                "quant_rabbit.strategy.forecast_persistence_tracker._history_has_cycle_pair",
                side_effect=stale_precheck,
            ):
                threads = [threading.Thread(target=worker) for _ in range(thread_count)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)

            self.assertEqual(errors, [])
            self.assertEqual(results.count(True), 1)
            rows = (root / "forecast_history.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)

    def test_distinct_cycle_flips_still_recommend_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for idx in range(3):
                record_forecast(
                    _forecast("EUR_USD", "DOWN"),
                    data_root=root,
                    cycle_id=f"cycle-{idx}",
                )

            verdict = assess_position(
                trade_id="t1",
                pair="EUR_USD",
                side="LONG",
                data_root=root,
            )

            self.assertEqual(verdict.verdict, "RECOMMEND_CLOSE")
            self.assertIn("flipped to DOWN", verdict.reason)

    def test_low_confidence_flips_do_not_recommend_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for idx in range(3):
                record_forecast(
                    _forecast("EUR_USD", "DOWN", confidence=0.21),
                    data_root=root,
                    cycle_id=f"cycle-{idx}",
                )

            verdict = assess_position(
                trade_id="t1",
                pair="EUR_USD",
                side="LONG",
                data_root=root,
            )

            self.assertEqual(verdict.verdict, "HOLD")
            self.assertIn("mixed forecast history", verdict.reason)

    def test_stale_forecast_history_cannot_recommend_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = datetime(2026, 5, 21, 0, 0, tzinfo=timezone.utc)
            for idx in range(5):
                record_forecast(
                    _forecast("EUR_USD", "UNCLEAR", confidence=0.2),
                    data_root=root,
                    now=base + timedelta(minutes=idx),
                    cycle_id=f"cycle-{idx}",
                )

            verdict = assess_position(
                trade_id="t1",
                pair="EUR_USD",
                side="LONG",
                data_root=root,
                fresh_after_utc=base + timedelta(hours=1),
            )

            self.assertEqual(verdict.verdict, "HOLD")
            self.assertIn("stale forecast history", verdict.reason)

    def test_stale_gap_does_not_stitch_old_unclear_run_to_fresh_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = datetime(2026, 5, 18, 0, 0, tzinfo=timezone.utc)
            for idx in range(5):
                record_forecast(
                    _forecast("EUR_USD", "UNCLEAR", confidence=0.2),
                    data_root=root,
                    now=base + timedelta(minutes=idx),
                    cycle_id=f"old-cycle-{idx}",
                )
            fresh = base + timedelta(days=11)
            record_forecast(
                _forecast("EUR_USD", "UNCLEAR", confidence=0.2),
                data_root=root,
                now=fresh,
                cycle_id="fresh-cycle",
            )

            verdict = assess_position(
                trade_id="t1",
                pair="EUR_USD",
                side="LONG",
                data_root=root,
                fresh_after_utc=fresh - timedelta(minutes=1),
            )

            self.assertEqual(verdict.verdict, "HOLD")
            self.assertEqual(verdict.last_n_directions, ("UNCLEAR",))
            self.assertIn("mixed forecast history", verdict.reason)

    def test_record_forecast_persists_audit_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecast = _forecast("EUR_USD", "UP", 0.67)
            forecast.current_price = 1.1000
            forecast.raw_confidence = 0.8
            forecast.calibration_multiplier = 0.84
            forecast.up_score = 42.0
            forecast.down_score = 12.0
            forecast.range_score = 4.0
            forecast.drivers_for = ("M15 BOS_UP",)
            forecast.drivers_against = ("H4 TREND_DOWN",)
            forecast.rationale_summary = "UP=42 DOWN=12"

            record_forecast(forecast, data_root=root, cycle_id="cycle-1")

            row = json.loads((root / "forecast_history.jsonl").read_text())
            self.assertEqual(row["cycle_id"], "cycle-1")
            self.assertEqual(row["current_price"], 1.1)
            self.assertEqual(row["drivers_for"], ["M15 BOS_UP"])
            self.assertEqual(row["drivers_against"], ["H4 TREND_DOWN"])
            self.assertEqual(row["up_score"], 42.0)


if __name__ == "__main__":
    unittest.main()
