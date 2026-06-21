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

    def test_contrarian_row_replays_opposite_bidask_entry_and_exit(self) -> None:
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
        scored = replay._score_one(
            row,
            [
                _candle("2026-06-19T00:00:00", bid_o=1.1000, bid_h=1.1001, bid_l=1.0999, bid_c=1.1000, ask_o=1.1002, ask_h=1.1003, ask_l=1.1001, ask_c=1.1002),
                _candle("2026-06-19T00:00:05", bid_o=1.0997, bid_h=1.0998, bid_l=1.0995, bid_c=1.0996, ask_o=1.0999, ask_h=1.1000, ask_l=1.0997, ask_c=1.0998),
            ],
        )

        contrarian = replay._contrarian_row(scored)

        self.assertIsNotNone(contrarian)
        self.assertEqual(contrarian["forecast_direction"], "UP")
        self.assertEqual(contrarian["direction"], "DOWN")
        self.assertTrue(contrarian["contrarian"])
        self.assertAlmostEqual(contrarian["entry_price"], 1.1000)
        self.assertAlmostEqual(contrarian["final_pips"], 2.0)
        self.assertTrue(contrarian["final_direction_hit"])

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

    def test_precision_rules_are_selected_from_pair_direction_segments(self) -> None:
        rules = replay._bidask_precision_rules(
            [
                {
                    "pair": "GBP_USD",
                    "direction": "DOWN",
                    "n": 48,
                    "summary": {
                        "hit_rate": 0.72,
                        "avg_final_pips": 1.8,
                        "median_final_pips": 1.2,
                        "avg_mfe_pips": 5.3,
                        "avg_mae_pips": 3.9,
                    },
                    "best_exit": {
                        "take_profit_pips": 5.0,
                        "stop_loss_pips": 7.0,
                        "avg_realized_pips": 2.1,
                        "win_rate": 0.69,
                        "profit_factor": 2.4,
                    },
                },
                {
                    "pair": "AUD_JPY",
                    "direction": "UP",
                    "n": 124,
                    "summary": {
                        "hit_rate": 0.20,
                        "avg_final_pips": -6.7,
                        "median_final_pips": -4.7,
                        "avg_mfe_pips": 3.8,
                        "avg_mae_pips": 14.4,
                    },
                    "best_exit": {
                        "take_profit_pips": 2.0,
                        "stop_loss_pips": 2.0,
                        "avg_realized_pips": -2.0,
                        "win_rate": 0.0,
                        "profit_factor": 0.0,
                    },
                },
                {
                    "pair": "EUR_JPY",
                    "direction": "DOWN",
                    "n": 7,
                    "summary": {
                        "hit_rate": 0.86,
                        "avg_final_pips": 8.0,
                        "median_final_pips": 7.0,
                        "avg_mfe_pips": 10.0,
                        "avg_mae_pips": 2.0,
                    },
                    "best_exit": {
                        "take_profit_pips": 5.0,
                        "stop_loss_pips": 7.0,
                        "avg_realized_pips": 3.0,
                        "win_rate": 0.86,
                        "profit_factor": 4.0,
                    },
                },
            ],
            granularity="S5",
            audit_report="unit.json",
            edge_min_samples=30,
            edge_min_directional_hit_rate=0.60,
            edge_min_avg_final_pips=0.0,
            edge_min_avg_realized_pips=0.5,
            edge_min_win_rate=0.55,
            edge_min_profit_factor=1.5,
            negative_min_samples=30,
            negative_max_directional_hit_rate=0.45,
            negative_max_avg_final_pips=0.0,
            negative_max_avg_realized_pips=-0.5,
            negative_max_win_rate=0.40,
            negative_max_profit_factor=0.75,
        )

        self.assertEqual(
            [rule["name"] for rule in rules["edge_rules"]],
            ["GBP_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7"],
        )
        self.assertEqual(rules["edge_rules"][0]["min_target_pips"], 4.8)
        self.assertEqual(
            [rule["name"] for rule in rules["negative_rules"]],
            ["AUD_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY"],
        )
        self.assertEqual(rules["negative_rules"][0]["side"], "LONG")
        self.assertEqual(rules["rejected_sampled_segments"], [])

    def test_precision_rules_select_contrarian_edge_from_losing_forecast_bucket(self) -> None:
        rules = replay._bidask_precision_rules(
            [],
            contrarian_segment_rows=[
                {
                    "pair": "AUD_JPY",
                    "forecast_direction": "UP",
                    "direction": "DOWN",
                    "n": 124,
                    "source_summary": {
                        "hit_rate": 0.20,
                        "avg_final_pips": -6.7,
                        "avg_mfe_pips": 3.8,
                        "avg_mae_pips": 14.4,
                    },
                    "summary": {
                        "hit_rate": 0.76,
                        "avg_final_pips": 5.8,
                        "median_final_pips": 4.1,
                        "avg_mfe_pips": 12.0,
                        "avg_mae_pips": 4.5,
                    },
                    "best_exit": {
                        "take_profit_pips": 5.0,
                        "stop_loss_pips": 7.0,
                        "avg_realized_pips": 2.4,
                        "win_rate": 0.70,
                        "profit_factor": 2.5,
                    },
                }
            ],
            granularity="S5",
            audit_report="unit.json",
            edge_min_samples=30,
            edge_min_directional_hit_rate=0.60,
            edge_min_avg_final_pips=0.0,
            edge_min_avg_realized_pips=0.5,
            edge_min_win_rate=0.55,
            edge_min_profit_factor=1.5,
            negative_min_samples=30,
            negative_max_directional_hit_rate=0.45,
            negative_max_avg_final_pips=0.0,
            negative_max_avg_realized_pips=-0.5,
            negative_max_win_rate=0.40,
            negative_max_profit_factor=0.75,
        )

        self.assertEqual(
            [rule["name"] for rule in rules["contrarian_edge_rules"]],
            ["AUD_JPY_UP_FADE_TO_DOWN_S5_BIDASK_CONTRARIAN_HARVEST_TP5_SL7"],
        )
        rule = rules["contrarian_edge_rules"][0]
        self.assertEqual(rule["side"], "SHORT")
        self.assertEqual(rule["faded_direction"], "UP")
        self.assertEqual(rule["direction"], "DOWN")
        self.assertEqual(rule["source_directional_hit_rate"], 0.2)
        self.assertTrue(rule["contrarian_edge"])

    def test_missing_price_window_groups_publish_fetch_windows(self) -> None:
        rows = [
            replay.ForecastRow(
                source_index=1,
                timestamp_utc=datetime(2026, 6, 17, 7, 21, tzinfo=timezone.utc),
                pair="AUD_JPY",
                direction="UP",
                confidence=0.7,
                current_price=None,
                target_price=None,
                invalidation_price=None,
                horizon_min=60,
                cycle_id=None,
            ),
            replay.ForecastRow(
                source_index=2,
                timestamp_utc=datetime(2026, 6, 17, 14, 4, tzinfo=timezone.utc),
                pair="EUR_USD",
                direction="DOWN",
                confidence=0.8,
                current_price=None,
                target_price=None,
                invalidation_price=None,
                horizon_min=240,
                cycle_id=None,
            ),
        ]

        groups = replay._missing_price_window_groups(rows)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["date"], "2026-06-17")
        self.assertEqual(groups[0]["count"], 2)
        self.assertEqual(groups[0]["needed_from_utc"], "2026-06-17T07:16:00Z")
        self.assertEqual(groups[0]["needed_to_utc"], "2026-06-17T18:09:00Z")
        self.assertEqual(groups[0]["pairs"], ["AUD_JPY", "EUR_USD"])
        self.assertEqual(groups[0]["pair_directions"], ["AUD_JPY:UP", "EUR_USD:DOWN"])

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
