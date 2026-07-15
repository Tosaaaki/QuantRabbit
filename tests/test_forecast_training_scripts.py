from __future__ import annotations

import importlib.util
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = _load("train_forecast_orientation_model")
horizons = _load("train_forecast_horizon_models")
replay = sys.modules["oanda_history_replay_validate"]


class ForecastTrainingScriptsTest(unittest.TestCase):
    def test_s5_quotes_downsample_to_executable_m1_ohlc(self) -> None:
        first = replay.QuoteCandle(
            timestamp_utc=datetime(2026, 7, 1, 10, 0, 0, tzinfo=timezone.utc),
            pair="EUR_USD",
            bid=replay.Ohlc(1.1000, 1.1003, 1.0999, 1.1002),
            ask=replay.Ohlc(1.1001, 1.1004, 1.1000, 1.1003),
        )
        second = replay.QuoteCandle(
            timestamp_utc=datetime(2026, 7, 1, 10, 0, 5, tzinfo=timezone.utc),
            pair="EUR_USD",
            bid=replay.Ohlc(1.1002, 1.1005, 1.1001, 1.1004),
            ask=replay.Ohlc(1.1003, 1.1006, 1.1002, 1.1005),
        )

        result = train._aggregate_quote_minute(
            [second, first],
            timestamp_utc=datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(result.bid, replay.Ohlc(1.1000, 1.1005, 1.0999, 1.1004))
        self.assertEqual(result.ask, replay.Ohlc(1.1001, 1.1006, 1.1000, 1.1005))

    def test_fixed_horizon_list_is_predeclared_unique_and_bounded(self) -> None:
        self.assertEqual(
            horizons._parse_horizons("5,15,15,60,240"),
            [5.0, 15.0, 60.0, 240.0],
        )
        with self.assertRaises(ValueError):
            horizons._parse_horizons("60")
        with self.assertRaises(ValueError):
            horizons._parse_horizons("5,1441")


if __name__ == "__main__":
    unittest.main()
