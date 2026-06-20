from __future__ import annotations

import importlib.util
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "oanda_history_replay_validate.py"
    spec = importlib.util.spec_from_file_location("oanda_history_replay_validate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = _load_module()


def _candle(ts: str, *, bid_o: float, bid_h: float, bid_l: float, bid_c: float, ask_o: float, ask_h: float, ask_l: float, ask_c: float):
    return replay.QuoteCandle(
        timestamp_utc=datetime.fromisoformat(ts).replace(tzinfo=timezone.utc),
        pair="EUR_USD",
        bid=replay.Ohlc(bid_o, bid_h, bid_l, bid_c),
        ask=replay.Ohlc(ask_o, ask_h, ask_l, ask_c),
    )


class OandaHistoryReplayValidateTest(unittest.TestCase):
    def test_up_forecast_uses_ask_entry_and_bid_exit(self) -> None:
        row = replay.ForecastRow(
            source_index=1,
            timestamp_utc=datetime(2026, 6, 19, tzinfo=timezone.utc),
            pair="EUR_USD",
            direction="UP",
            confidence=0.7,
            current_price=None,
            target_price=None,
            invalidation_price=None,
            horizon_min=1,
            cycle_id=None,
        )
        result = replay._score_one(
            row,
            [
                _candle("2026-06-19T00:00:00", bid_o=1.1000, bid_h=1.1001, bid_l=1.0999, bid_c=1.1000, ask_o=1.1002, ask_h=1.1003, ask_l=1.1001, ask_c=1.1002),
                _candle("2026-06-19T00:00:05", bid_o=1.1003, bid_h=1.1005, bid_l=1.1002, bid_c=1.1004, ask_o=1.1005, ask_h=1.1007, ask_l=1.1004, ask_c=1.1006),
            ],
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["entry_price"], 1.1002)
        self.assertAlmostEqual(result["final_pips"], 2.0)
        self.assertTrue(result["final_direction_hit"])

    def test_down_forecast_uses_bid_entry_and_ask_exit(self) -> None:
        row = replay.ForecastRow(
            source_index=1,
            timestamp_utc=datetime(2026, 6, 19, tzinfo=timezone.utc),
            pair="EUR_USD",
            direction="DOWN",
            confidence=0.7,
            current_price=None,
            target_price=None,
            invalidation_price=None,
            horizon_min=1,
            cycle_id=None,
        )
        result = replay._score_one(
            row,
            [
                _candle("2026-06-19T00:00:00", bid_o=1.1000, bid_h=1.1001, bid_l=1.0999, bid_c=1.1000, ask_o=1.1002, ask_h=1.1003, ask_l=1.1001, ask_c=1.1002),
                _candle("2026-06-19T00:00:05", bid_o=1.0997, bid_h=1.0998, bid_l=1.0995, bid_c=1.0996, ask_o=1.0999, ask_h=1.1000, ask_l=1.0997, ask_c=1.0998),
            ],
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["entry_price"], 1.1000)
        self.assertAlmostEqual(result["final_pips"], 2.0)
        self.assertTrue(result["final_direction_hit"])

    def test_same_candle_tp_and_sl_counts_as_stop_first(self) -> None:
        row = {
            "pair": "EUR_USD",
            "direction": "UP",
            "entry_price": 1.1000,
            "final_pips": 0.0,
            "_window": [
                _candle("2026-06-19T00:00:00", bid_o=1.1000, bid_h=1.1003, bid_l=1.0998, bid_c=1.1000, ask_o=1.1002, ask_h=1.1005, ask_l=1.1000, ask_c=1.1002)
            ],
        }

        result = replay._simulate_exit(row, take_profit_pips=2.0, stop_loss_pips=2.0)

        self.assertEqual(result, {"pips": -2.0, "reason": "SL"})

    def test_segment_exit_grids_keep_pair_direction_best_exit(self) -> None:
        row = {
            "pair": "EUR_USD",
            "direction": "DOWN",
            "entry_price": 1.1000,
            "final_pips": 3.0,
            "final_direction_hit": True,
            "mfe_pips": 5.5,
            "mae_pips": 1.0,
            "_window": [
                _candle("2026-06-19T00:00:00", bid_o=1.1000, bid_h=1.1001, bid_l=1.0999, bid_c=1.1000, ask_o=1.1002, ask_h=1.1003, ask_l=1.1001, ask_c=1.1002),
                _candle("2026-06-19T00:00:05", bid_o=1.0995, bid_h=1.0996, bid_l=1.0993, bid_c=1.0995, ask_o=1.0997, ask_h=1.0998, ask_l=1.0995, ask_c=1.0997),
            ],
        }

        segments = replay._segment_exit_grids(
            [row],
            ("pair", "direction"),
            tp_grid=(5.0,),
            sl_grid=(7.0,),
            min_n=1,
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["pair"], "EUR_USD")
        self.assertEqual(segments[0]["direction"], "DOWN")
        self.assertEqual(segments[0]["best_exit"]["take_profit_pips"], 5.0)
        self.assertEqual(segments[0]["best_exit"]["stop_loss_pips"], 7.0)
        self.assertEqual(segments[0]["best_exit"]["avg_realized_pips"], 5.0)

    def test_target_not_on_reward_side_is_not_counted_as_touch(self) -> None:
        row = replay.ForecastRow(
            source_index=1,
            timestamp_utc=datetime(2026, 6, 19, tzinfo=timezone.utc),
            pair="EUR_USD",
            direction="UP",
            confidence=0.7,
            current_price=None,
            target_price=1.0990,
            invalidation_price=1.0980,
            horizon_min=1,
            cycle_id=None,
        )

        result = replay._score_one(
            row,
            [
                _candle("2026-06-19T00:00:00", bid_o=1.1000, bid_h=1.1001, bid_l=1.0999, bid_c=1.1000, ask_o=1.1002, ask_h=1.1003, ask_l=1.1001, ask_c=1.1002),
                _candle("2026-06-19T00:00:05", bid_o=1.1000, bid_h=1.1005, bid_l=1.0990, bid_c=1.1001, ask_o=1.1002, ask_h=1.1007, ask_l=1.0992, ask_c=1.1003),
            ],
        )

        self.assertIsNotNone(result)
        self.assertFalse(result["target_reward_side"])
        self.assertIsNone(result["target_touch"])
        self.assertIsNone(result["target_before_invalidation"])


if __name__ == "__main__":
    unittest.main()
