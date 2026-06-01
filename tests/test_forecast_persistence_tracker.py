from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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
