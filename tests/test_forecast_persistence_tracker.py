from __future__ import annotations

import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
