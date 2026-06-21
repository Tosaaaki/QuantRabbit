from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
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
    def test_history_dirs_prefers_multi_month_suite_over_short_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "logs" / "replay" / "oanda_history"
            short_run = history / "20260620T153021Z"
            long_run = history / "months_g1" / "20260621T020921Z"
            short_run.mkdir(parents=True)
            long_run.mkdir(parents=True)
            (history / "latest_summary.json").write_text(
                json.dumps(
                    {
                        "output_dir": str(short_run),
                        "granularities": ["S5"],
                        "window": {
                            "from": "2026-06-08T00:00:00Z",
                            "to": "2026-06-09T00:00:00Z",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (short_run / "summary.json").write_text(
                json.dumps(
                    {
                        "output_dir": str(short_run),
                        "granularities": ["S5"],
                        "window": {
                            "from": "2026-06-08T00:00:00Z",
                            "to": "2026-06-09T00:00:00Z",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (long_run / "summary.json").write_text(
                json.dumps(
                    {
                        "output_dir": str(long_run),
                        "granularities": ["S5", "M5"],
                        "window": {
                            "from": "2026-03-16T00:00:00Z",
                            "to": "2026-06-20T00:00:00Z",
                        },
                    }
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                dirs = replay._history_dirs(None, granularity="S5", auto_min_days=30.0)
            finally:
                os.chdir(previous)

        self.assertEqual(dirs, [long_run])

    def test_history_dirs_discovers_orphan_multi_month_run_from_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "logs" / "replay" / "oanda_history"
            short_run = history / "20260620T153021Z"
            orphan_run = history / "20260621T015218Z"
            candle_dir = orphan_run / "GBP_USD"
            short_run.mkdir(parents=True)
            candle_dir.mkdir(parents=True)
            (history / "latest_summary.json").write_text(
                json.dumps({"output_dir": str(short_run), "granularities": ["S5"]}),
                encoding="utf-8",
            )
            (candle_dir / "GBP_USD_S5_BA_20260316T000000Z_20260620T000000Z.jsonl").write_text(
                "",
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                dirs = [path.resolve() for path in replay._history_dirs(None, granularity="S5", auto_min_days=30.0)]
            finally:
                os.chdir(previous)

        self.assertEqual(dirs, [orphan_run.resolve()])

    def test_history_dirs_falls_back_to_latest_when_no_multi_month_suite_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "logs" / "replay" / "oanda_history"
            short_run = history / "20260620T153021Z"
            short_run.mkdir(parents=True)
            (history / "latest_summary.json").write_text(
                json.dumps({"output_dir": str(short_run), "granularities": ["S5"]}),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                dirs = replay._history_dirs(None, granularity="S5", auto_min_days=30.0)
            finally:
                os.chdir(previous)

        self.assertEqual(dirs, [short_run])

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
            "timestamp_utc": "2026-06-19T00:00:00Z",
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
        self.assertEqual(segments[0]["daily_stability"]["active_days"], 1)

    def test_daily_exit_stability_flags_multi_day_distribution(self) -> None:
        def row(ts: str, *, high: float, low: float) -> dict:
            return {
                "timestamp_utc": ts,
                "pair": "EUR_USD",
                "direction": "UP",
                "entry_price": 1.1000,
                "final_pips": 0.0,
                "_window": [
                    _candle(ts.replace("Z", ""), bid_o=1.1000, bid_h=high, bid_l=low, bid_c=1.1000, ask_o=1.1002, ask_h=high + 0.0002, ask_l=low + 0.0002, ask_c=1.1002)
                ],
            }

        daily = replay._daily_exit_stability(
            [
                row("2026-06-18T15:00:00Z", high=1.1006, low=1.0999),
                row("2026-06-18T16:00:00Z", high=1.1006, low=1.0999),
                row("2026-06-19T15:00:00Z", high=1.1001, low=1.0992),
                row("2026-06-20T15:00:00Z", high=1.1006, low=1.0999),
            ],
            take_profit_pips=5.0,
            stop_loss_pips=7.0,
        )

        self.assertEqual(daily["active_days"], 3)
        self.assertEqual(daily["positive_days"], 2)
        self.assertEqual(daily["negative_days"], 1)
        self.assertEqual(daily["max_daily_sample_share"], 0.5)
        self.assertEqual(
            replay._daily_stability_status(
                daily,
                min_active_days=3,
                max_daily_sample_share=0.70,
                min_positive_day_rate=2.0 / 3.0,
            ),
            "DAILY_STABLE",
        )

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
                    "daily_stability": {
                        "campaign_timezone": "Asia/Tokyo",
                        "active_days": 3,
                        "first_day": "2026-06-17",
                        "last_day": "2026-06-19",
                        "min_daily_samples": 8,
                        "max_daily_samples": 20,
                        "avg_daily_samples": 16.0,
                        "max_daily_sample_share": 0.4167,
                        "positive_days": 3,
                        "negative_days": 0,
                        "flat_days": 0,
                        "positive_day_rate": 1.0,
                        "avg_daily_realized_pips": 33.6,
                        "worst_daily_realized_pips": 11.2,
                        "best_daily_realized_pips": 58.8,
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
            [rule["name"] for rule in rules["daily_stable_edge_rules"]],
            ["GBP_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7"],
        )
        self.assertEqual(rules["edge_rules"][0]["daily_stability_status"], "DAILY_STABLE")
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

    def test_forecast_sample_coverage_exposes_under_sampled_gbp_pair_direction(self) -> None:
        rows = [
            replay.ForecastRow(
                source_index=idx,
                timestamp_utc=datetime(2026, 6, 17, idx, tzinfo=timezone.utc),
                pair="GBP_USD",
                direction="DOWN",
                confidence=0.7,
                current_price=None,
                target_price=None,
                invalidation_price=None,
                horizon_min=60,
                cycle_id=None,
            )
            for idx in range(2)
        ]
        results = [
            {
                "timestamp_utc": "2026-06-17T00:00:00Z",
                "pair": "GBP_USD",
                "direction": "DOWN",
            }
        ]

        coverage = replay._forecast_sample_coverage(
            rows,
            results,
            min_directional_samples=30,
            min_active_days=3,
        )

        self.assertEqual(coverage["pair_count"], 1)
        self.assertEqual(coverage["pair_direction_count"], 1)
        gap = coverage["under_sampled_pair_directions"][0]
        self.assertEqual(gap["pair"], "GBP_USD")
        self.assertEqual(gap["direction"], "DOWN")
        self.assertEqual(gap["forecast_samples"], 2)
        self.assertEqual(gap["evaluated_samples"], 1)
        self.assertEqual(gap["missing_price_truth_samples"], 1)
        self.assertEqual(gap["missing_evaluated_samples"], 29)
        self.assertEqual(gap["missing_active_days"], 2)
        self.assertIn("INSUFFICIENT_EVALUATED_SAMPLES", gap["coverage_gap_reasons"])
        self.assertIn("INSUFFICIENT_ACTIVE_DAYS", gap["coverage_gap_reasons"])
        self.assertIn("PRICE_TRUTH_WINDOW_MISSING", gap["coverage_gap_reasons"])

    def test_load_candles_filters_to_forecast_truth_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "history"
            pair_dir = history / "EUR_USD"
            pair_dir.mkdir(parents=True)
            path = pair_dir / "EUR_USD_S5_BA_20260619T000000Z_20260619T010000Z.jsonl"
            rows = [
                {
                    "pair": "EUR_USD",
                    "granularity": "S5",
                    "time": "2026-06-19T00:00:00Z",
                    "bid": {"o": "1.1000", "h": "1.1001", "l": "1.0999", "c": "1.1000"},
                    "ask": {"o": "1.1002", "h": "1.1003", "l": "1.1001", "c": "1.1002"},
                },
                {
                    "pair": "EUR_USD",
                    "granularity": "S5",
                    "time": "2026-06-19T00:30:00Z",
                    "bid": {"o": "1.1010", "h": "1.1011", "l": "1.1009", "c": "1.1010"},
                    "ask": {"o": "1.1012", "h": "1.1013", "l": "1.1011", "c": "1.1012"},
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            candles, stats = replay._load_candles(
                [history],
                granularity="S5",
                windows_by_pair={
                    "EUR_USD": [
                        (
                            datetime(2026, 6, 18, 23, 59, tzinfo=timezone.utc),
                            datetime(2026, 6, 19, 0, 1, tzinfo=timezone.utc),
                        )
                    ]
                },
            )

        self.assertEqual(stats["history_raw_rows"], 2)
        self.assertEqual(stats["history_filtered_rows"], 1)
        self.assertEqual(stats["history_candles"], 1)
        self.assertEqual([c.timestamp_utc.isoformat() for c in candles["EUR_USD"]], ["2026-06-19T00:00:00+00:00"])

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
