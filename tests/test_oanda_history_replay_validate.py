from __future__ import annotations

import importlib.util
import gzip
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
    def test_history_dirs_keeps_multi_month_suite_and_windowed_overlays(self) -> None:
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

        self.assertEqual(dirs, [long_run, short_run])

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

    def test_explicit_history_parent_discovers_nested_multi_month_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "oanda_history_s5"
            run = history / "20260622T155928Z"
            candle_dir = run / "AUD_JPY"
            candle_dir.mkdir(parents=True)
            (run / "summary.json").write_text(
                json.dumps(
                    {
                        "output_dir": str(run),
                        "granularities": ["S5"],
                        "window": {
                            "from": "2026-02-22T15:59:19Z",
                            "to": "2026-06-22T15:59:19Z",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (candle_dir / "AUD_JPY_S5_BA_20260222T155919Z_20260622T155919Z.jsonl").write_text(
                "",
                encoding="utf-8",
            )

            dirs = replay._history_dirs([history], granularity="S5", auto_min_days=30.0)

        self.assertEqual(dirs, [run])

    def test_explicit_history_parent_discovers_windowed_short_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "oanda_history_s5"
            run_a = history / "20260622T155928Z"
            run_b = history / "20260622T160015Z"
            other_granularity = history / "20260622T160111Z"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            other_granularity.mkdir(parents=True)
            (run_a / "summary.json").write_text(
                json.dumps(
                    {
                        "output_dir": str(run_a),
                        "granularities": ["S5"],
                        "window": {
                            "from": "2026-06-17T07:16:00Z",
                            "to": "2026-06-17T18:09:00Z",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_b / "summary.json").write_text(
                json.dumps(
                    {
                        "output_dir": str(run_b),
                        "granularities": ["S5"],
                        "window": {
                            "from": "2026-06-18T01:00:00Z",
                            "to": "2026-06-18T02:00:00Z",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (other_granularity / "summary.json").write_text(
                json.dumps({"output_dir": str(other_granularity), "granularities": ["M5"]}),
                encoding="utf-8",
            )
            (history / "latest_summary.json").write_text(
                json.dumps({"output_dir": str(run_b), "granularities": ["S5"]}),
                encoding="utf-8",
            )

            dirs = replay._history_dirs([history], granularity="S5", auto_min_days=30.0)

        self.assertEqual(dirs, [run_a, run_b])

    def test_explicit_history_parent_keeps_windowed_overlays_with_multi_month_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "oanda_history_s5"
            long_run = history / "20260622T155928Z"
            overlay_run = history / "20260622T160015Z"
            long_run.mkdir(parents=True)
            overlay_run.mkdir(parents=True)
            (long_run / "summary.json").write_text(
                json.dumps(
                    {
                        "output_dir": str(long_run),
                        "granularities": ["S5"],
                        "window": {
                            "from": "2026-02-22T15:59:19Z",
                            "to": "2026-06-22T15:59:19Z",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (overlay_run / "summary.json").write_text(
                json.dumps(
                    {
                        "output_dir": str(overlay_run),
                        "granularities": ["S5"],
                        "window": {
                            "from": "2026-06-22T00:42:48Z",
                            "to": "2026-06-22T21:02:28Z",
                        },
                    }
                ),
                encoding="utf-8",
            )

            dirs = replay._history_dirs([history], granularity="S5", auto_min_days=30.0)

        self.assertEqual(dirs, [long_run, overlay_run])

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
        self.assertEqual(rules["edge_rules"][0]["adoption_status"], "LIVE_GRADE_DAILY_STABLE")
        self.assertTrue(rules["edge_rules"][0]["live_grade"])
        self.assertEqual(
            [rule["name"] for rule in rules["negative_rules"]],
            ["AUD_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY"],
        )
        self.assertEqual(
            rules["negative_rules"][0]["adoption_status"],
            "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        )
        self.assertEqual(rules["negative_rules"][0]["side"], "LONG")
        self.assertEqual(rules["adoption_summary"]["live_grade_support_rules"], 1)
        self.assertEqual(rules["adoption_summary"]["negative_block_rules"], 1)
        self.assertEqual(rules["rejected_sampled_segments"], [])

    def test_precision_rules_mark_concentrated_edge_rank_only(self) -> None:
        rules = replay._bidask_precision_rules(
            [
                {
                    "pair": "EUR_USD",
                    "direction": "DOWN",
                    "n": 226,
                    "summary": {
                        "hit_rate": 0.74,
                        "avg_final_pips": 2.3,
                        "median_final_pips": 3.7,
                        "avg_mfe_pips": 5.1,
                        "avg_mae_pips": 4.3,
                    },
                    "best_exit": {
                        "take_profit_pips": 5.0,
                        "stop_loss_pips": 7.0,
                        "avg_realized_pips": 2.58,
                        "win_rate": 0.74,
                        "profit_factor": 3.34,
                    },
                    "daily_stability": {
                        "campaign_timezone": "Asia/Tokyo",
                        "active_days": 5,
                        "first_day": "2026-05-15",
                        "last_day": "2026-06-16",
                        "min_daily_samples": 1,
                        "max_daily_samples": 219,
                        "avg_daily_samples": 45.2,
                        "max_daily_sample_share": 0.969,
                        "positive_days": 2,
                        "negative_days": 3,
                        "flat_days": 0,
                        "positive_day_rate": 0.4,
                        "avg_daily_realized_pips": 116.76,
                        "worst_daily_realized_pips": -14.0,
                        "best_daily_realized_pips": 606.1,
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

        rule = rules["edge_rules"][0]
        self.assertEqual(rule["adoption_status"], "RANK_ONLY_NOT_DAILY_STABLE")
        self.assertFalse(rule["live_grade"])
        self.assertIn("DAILY_SAMPLE_CONCENTRATED", rule["adoption_blockers"])
        self.assertIn("NEEDS_LESS_DAILY_SAMPLE_CONCENTRATION", rule["adoption_blockers"])
        self.assertIn("NEEDS_HIGHER_POSITIVE_DAY_RATE", rule["adoption_blockers"])
        self.assertEqual(rule["daily_stability_gap"]["current_max_daily_sample_share"], 0.969)
        self.assertEqual(rules["daily_stable_edge_rules"], [])
        self.assertEqual(rules["adoption_summary"]["live_grade_support_rules"], 0)
        self.assertEqual(rules["adoption_summary"]["rank_only_support_rules"], 1)
        self.assertEqual(
            rules["adoption_summary"]["rank_only_blocker_counts"]["DAILY_SAMPLE_CONCENTRATED"],
            1,
        )

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
        self.assertEqual(rule["adoption_status"], "RANK_ONLY_NOT_DAILY_STABLE")
        self.assertFalse(rule["live_grade"])
        self.assertIn("INSUFFICIENT_ACTIVE_DAYS", rule["adoption_blockers"])

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

    def test_saturday_no_market_windows_are_not_price_truth_missing(self) -> None:
        row = replay.ForecastRow(
            source_index=1,
            timestamp_utc=datetime(2026, 5, 16, 1, 4, tzinfo=timezone.utc),
            pair="AUD_JPY",
            direction="UP",
            confidence=0.8,
            current_price=None,
            target_price=None,
            invalidation_price=None,
            horizon_min=60,
            cycle_id=None,
        )

        results, score_stats, no_market_rows, pending_rows = replay._score_forecasts(
            [row],
            {},
            now_utc=datetime(2026, 6, 23, tzinfo=timezone.utc),
        )
        coverage = replay._forecast_sample_coverage(
            [row],
            results,
            unscorable_no_market_rows=no_market_rows,
            pending_future_truth_rows=pending_rows,
            min_directional_samples=1,
            min_active_days=1,
        )
        truth = replay._price_truth_coverage(
            load_stats={"raw_directional_rows": 1, "deduped_directional_rows": 1},
            candle_stats={"history_files": 0, "history_candles": 0},
            score_stats=score_stats,
            sample_coverage=coverage,
            granularity="S5",
            edge_min_samples=1,
        )

        self.assertEqual(results, [])
        self.assertEqual(score_stats["missing_price_window_groups"], [])
        self.assertEqual(score_stats["unscorable_no_market_rows"], 1)
        self.assertEqual(score_stats["unscorable_no_market_window_groups"][0]["pairs"], ["AUD_JPY"])
        pair_row = coverage["pairs"][0]
        self.assertEqual(pair_row["unscorable_no_market_samples"], 1)
        self.assertEqual(pair_row["missing_price_truth_samples"], 0)
        gap = coverage["under_sampled_pair_directions"][0]
        self.assertIn("NO_MARKET_SESSION_UNSCORABLE", gap["coverage_gap_reasons"])
        self.assertNotIn("PRICE_TRUTH_WINDOW_MISSING", gap["coverage_gap_reasons"])
        self.assertEqual(truth["status"], "NO_SCORABLE_MARKET_FORECAST_ROWS")
        self.assertEqual(truth["missing_price_truth_samples"], 0)
        self.assertIn("FORECAST_ROWS_DURING_BROKER_NO_MARKET_WINDOW", truth["warnings"])

    def test_weekend_close_crossing_windows_are_not_price_truth_missing(self) -> None:
        rows = [
            replay.ForecastRow(
                source_index=1,
                timestamp_utc=datetime(2026, 5, 22, 21, 5, tzinfo=timezone.utc),
                pair="GBP_USD",
                direction="DOWN",
                confidence=0.8,
                current_price=None,
                target_price=None,
                invalidation_price=None,
                horizon_min=240,
                cycle_id=None,
            ),
            replay.ForecastRow(
                source_index=2,
                timestamp_utc=datetime(2026, 6, 7, 18, 34, tzinfo=timezone.utc),
                pair="EUR_JPY",
                direction="UP",
                confidence=0.8,
                current_price=None,
                target_price=None,
                invalidation_price=None,
                horizon_min=60,
                cycle_id=None,
            ),
            replay.ForecastRow(
                source_index=3,
                timestamp_utc=datetime(2026, 6, 5, 20, 54, tzinfo=timezone.utc),
                pair="USD_JPY",
                direction="UP",
                confidence=0.8,
                current_price=None,
                target_price=None,
                invalidation_price=None,
                horizon_min=15,
                cycle_id=None,
            ),
        ]

        results, score_stats, no_market_rows, pending_rows = replay._score_forecasts(
            rows,
            {},
            now_utc=datetime(2026, 6, 23, tzinfo=timezone.utc),
        )
        coverage = replay._forecast_sample_coverage(
            rows,
            results,
            unscorable_no_market_rows=no_market_rows,
            pending_future_truth_rows=pending_rows,
            min_directional_samples=1,
            min_active_days=1,
        )
        truth = replay._price_truth_coverage(
            load_stats={"raw_directional_rows": 3, "deduped_directional_rows": 3},
            candle_stats={"history_files": 0, "history_candles": 0},
            score_stats=score_stats,
            sample_coverage=coverage,
            granularity="S5",
            edge_min_samples=1,
            now_utc=datetime(2026, 6, 23, tzinfo=timezone.utc),
        )

        self.assertEqual(results, [])
        self.assertEqual(score_stats["missing_price_window_groups"], [])
        self.assertEqual(score_stats["unscorable_no_market_rows"], 3)
        self.assertEqual(
            sorted(group["pairs"][0] for group in score_stats["unscorable_no_market_window_groups"]),
            ["EUR_JPY", "GBP_USD", "USD_JPY"],
        )
        self.assertEqual(truth["missing_price_truth_samples"], 0)
        self.assertEqual(truth["history_fetch_command_count"], 0)
        self.assertNotIn("FETCH_MISSING_PRICE_TRUTH", truth["blockers"])

    def test_future_horizon_is_pending_not_price_truth_missing(self) -> None:
        row = replay.ForecastRow(
            source_index=1,
            timestamp_utc=datetime(2026, 6, 23, 16, 34, tzinfo=timezone.utc),
            pair="AUD_CAD",
            direction="UP",
            confidence=0.8,
            current_price=None,
            target_price=None,
            invalidation_price=None,
            horizon_min=90,
            cycle_id=None,
        )
        now = datetime(2026, 6, 23, 16, 59, tzinfo=timezone.utc)

        results, score_stats, no_market_rows, pending_rows = replay._score_forecasts(
            [row],
            {},
            now_utc=now,
        )
        coverage = replay._forecast_sample_coverage(
            [row],
            results,
            unscorable_no_market_rows=no_market_rows,
            pending_future_truth_rows=pending_rows,
            min_directional_samples=1,
            min_active_days=1,
        )
        truth = replay._price_truth_coverage(
            load_stats={"raw_directional_rows": 1, "deduped_directional_rows": 1},
            candle_stats={"history_files": 0, "history_candles": 0},
            score_stats=score_stats,
            sample_coverage=coverage,
            granularity="S5",
            edge_min_samples=1,
            now_utc=now,
        )

        self.assertEqual(results, [])
        self.assertEqual(score_stats["missing_price_window_groups"], [])
        self.assertEqual(score_stats["pending_future_truth_rows"], 1)
        self.assertEqual(score_stats["pending_future_truth_window_groups"][0]["pairs"], ["AUD_CAD"])
        gap = coverage["under_sampled_pair_directions"][0]
        self.assertIn("PENDING_FUTURE_TRUTH_WINDOW", gap["coverage_gap_reasons"])
        self.assertNotIn("PRICE_TRUTH_WINDOW_MISSING", gap["coverage_gap_reasons"])
        self.assertEqual(truth["missing_price_truth_samples"], 0)
        self.assertEqual(truth["history_fetch_command_count"], 0)
        self.assertIn("FORECAST_ROWS_WITH_PENDING_FUTURE_TRUTH_WINDOW", truth["warnings"])
        self.assertNotIn("FETCH_MISSING_PRICE_TRUTH", truth["blockers"])

    def test_price_truth_coverage_blocks_empty_history_validation(self) -> None:
        truth = replay._price_truth_coverage(
            load_stats={"raw_directional_rows": 2, "deduped_directional_rows": 2},
            candle_stats={"history_files": 0, "history_candles": 0},
            score_stats={
                "evaluated_rows": 0,
                "missing_price_window_groups": [
                    {
                        "date": "2026-06-17",
                        "count": 2,
                        "needed_from_utc": "2026-06-17T07:16:00Z",
                        "needed_to_utc": "2026-06-17T18:09:00Z",
                        "pairs": ["AUD_JPY", "EUR_USD"],
                    }
                ],
            },
            sample_coverage={
                "pairs": [
                    {
                        "pair": "AUD_JPY",
                        "forecast_samples": 1,
                        "evaluated_samples": 0,
                        "missing_price_truth_samples": 1,
                    },
                    {
                        "pair": "EUR_USD",
                        "forecast_samples": 1,
                        "evaluated_samples": 0,
                        "missing_price_truth_samples": 1,
                    },
                ],
                "under_sampled_pair_directions": [
                    {
                        "pair": "AUD_JPY",
                        "direction": "UP",
                        "coverage_gap_reasons": ["PRICE_TRUTH_WINDOW_MISSING"],
                    }
                ],
            },
            granularity="S5",
            edge_min_samples=30,
        )

        self.assertEqual(truth["status"], "NO_PRICE_HISTORY_FILES")
        self.assertEqual(truth["adoption_level"], "NO_REPLAY_EVIDENCE")
        self.assertTrue(truth["candidate_rule_validation_blocked"])
        self.assertTrue(truth["global_currency_validation_blocked"])
        self.assertEqual(truth["missing_price_truth_samples"], 2)
        self.assertEqual(truth["missing_pairs"], ["AUD_JPY", "EUR_USD"])
        self.assertIn("DO_NOT_PROMOTE_PRECISION_RULE", truth["blockers"])
        self.assertIn("DO_NOT_CLAIM_ALL_CURRENCY_VALIDATION", truth["blockers"])
        self.assertIn("--pairs AUD_JPY,EUR_USD", truth["history_fetch_command"])
        self.assertIn("--from 2026-06-17T07:16:00Z", truth["history_fetch_command"])
        self.assertIn("--to 2026-06-17T18:09:00Z", truth["history_fetch_command"])

    def test_price_truth_coverage_marks_partial_history_rank_only(self) -> None:
        truth = replay._price_truth_coverage(
            load_stats={"raw_directional_rows": 40, "deduped_directional_rows": 40},
            candle_stats={"history_files": 1, "history_candles": 500},
            score_stats={
                "evaluated_rows": 35,
                "missing_price_window_groups": [
                    {
                        "date": "2026-06-18",
                        "count": 5,
                        "needed_from_utc": "2026-06-18T00:00:00Z",
                        "needed_to_utc": "2026-06-18T02:00:00Z",
                        "pairs": ["GBP_USD"],
                    }
                ],
            },
            sample_coverage={
                "pairs": [
                    {
                        "pair": "AUD_JPY",
                        "forecast_samples": 35,
                        "evaluated_samples": 35,
                        "missing_price_truth_samples": 0,
                    },
                    {
                        "pair": "GBP_USD",
                        "forecast_samples": 5,
                        "evaluated_samples": 0,
                        "missing_price_truth_samples": 5,
                    },
                ],
                "under_sampled_pair_directions": [
                    {
                        "pair": "GBP_USD",
                        "direction": "DOWN",
                        "coverage_gap_reasons": ["PRICE_TRUTH_WINDOW_MISSING"],
                    }
                ],
            },
            granularity="S5",
            edge_min_samples=30,
        )

        self.assertEqual(truth["status"], "PARTIAL_PRICE_TRUTH")
        self.assertEqual(truth["adoption_level"], "PAIR_LOCAL_RANK_ONLY")
        self.assertFalse(truth["candidate_rule_validation_blocked"])
        self.assertTrue(truth["global_currency_validation_blocked"])
        self.assertEqual(truth["missing_pairs"], ["GBP_USD"])
        self.assertEqual(truth["missing_pair_directions"], ["GBP_USD:DOWN"])
        self.assertNotIn("DO_NOT_PROMOTE_PRECISION_RULE", truth["blockers"])
        self.assertIn("DO_NOT_CLAIM_ALL_CURRENCY_VALIDATION", truth["blockers"])

    def test_price_truth_ok_still_blocks_all_currency_claim_when_samples_are_thin(self) -> None:
        truth = replay._price_truth_coverage(
            load_stats={"raw_directional_rows": 40, "deduped_directional_rows": 40},
            candle_stats={"history_files": 28, "history_candles": 5000},
            score_stats={
                "evaluated_rows": 40,
                "missing_price_window_groups": [],
            },
            sample_coverage={
                "pairs": [
                    {
                        "pair": "AUD_JPY",
                        "forecast_samples": 35,
                        "evaluated_samples": 35,
                        "missing_price_truth_samples": 0,
                    },
                    {
                        "pair": "GBP_USD",
                        "forecast_samples": 5,
                        "evaluated_samples": 5,
                        "missing_price_truth_samples": 0,
                    },
                ],
                "under_sampled_pair_directions": [
                    {
                        "pair": "GBP_USD",
                        "direction": "DOWN",
                        "missing_evaluated_samples": 25,
                        "coverage_gap_reasons": [
                            "INSUFFICIENT_EVALUATED_SAMPLES",
                            "INSUFFICIENT_ACTIVE_DAYS",
                        ],
                    }
                ],
            },
            granularity="S5",
            edge_min_samples=30,
        )

        self.assertEqual(truth["status"], "PRICE_TRUTH_OK")
        self.assertEqual(truth["adoption_level"], "PAIR_LOCAL_RANK_ONLY")
        self.assertFalse(truth["candidate_rule_validation_blocked"])
        self.assertTrue(truth["global_currency_validation_blocked"])
        self.assertEqual(truth["all_currency_sample_coverage_status"], "UNDER_SAMPLED")
        self.assertEqual(truth["under_sampled_pair_direction_count"], 1)
        self.assertEqual(truth["under_sampled_pair_directions"], ["GBP_USD:DOWN"])
        self.assertIn("DO_NOT_CLAIM_ALL_CURRENCY_VALIDATION", truth["blockers"])
        self.assertIn("COLLECT_MORE_FORECAST_SAMPLES", truth["blockers"])
        self.assertNotIn("FETCH_MISSING_PRICE_TRUTH", truth["blockers"])

    def test_history_fetch_command_clamps_future_window_to_now(self) -> None:
        command = replay._history_fetch_command(
            [
                {
                    "date": "2026-06-17",
                    "count": 1,
                    "needed_from_utc": "2026-06-17T10:00:00Z",
                    "needed_to_utc": "2026-06-17T14:00:00Z",
                    "pairs": ["USD_JPY"],
                }
            ],
            "S5",
            now_utc=datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc),
        )

        self.assertIsNotNone(command)
        self.assertIn("--pairs USD_JPY", command)
        self.assertIn("--from 2026-06-17T10:00:00Z", command)
        self.assertIn("--to 2026-06-17T12:00:00Z", command)

    def test_history_fetch_commands_preserve_windowed_missing_groups(self) -> None:
        commands = replay._history_fetch_commands(
            [
                {
                    "date": "2026-06-17",
                    "count": 2,
                    "needed_from_utc": "2026-06-17T07:16:00Z",
                    "needed_to_utc": "2026-06-17T18:09:00Z",
                    "pairs": ["EUR_USD", "AUD_JPY"],
                },
                {
                    "date": "2026-06-18",
                    "count": 1,
                    "needed_from_utc": "2026-06-18T01:00:00Z",
                    "needed_to_utc": "2026-06-18T02:00:00Z",
                    "pairs": ["GBP_USD"],
                },
            ],
            "S5",
            now_utc=datetime(2026, 6, 19, tzinfo=timezone.utc),
        )

        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0]["date"], "2026-06-17")
        self.assertEqual(commands[0]["forecast_rows_missing_truth"], 2)
        self.assertEqual(commands[0]["pairs"], ["AUD_JPY", "EUR_USD"])
        self.assertIn("--pairs AUD_JPY,EUR_USD", commands[0]["command"])
        self.assertIn("--from 2026-06-17T07:16:00Z", commands[0]["command"])
        self.assertIn("--to 2026-06-17T18:09:00Z", commands[0]["command"])
        self.assertEqual(commands[1]["date"], "2026-06-18")
        self.assertIn("--pairs GBP_USD", commands[1]["command"])
        self.assertNotIn("AUD_JPY", commands[1]["command"])

    def test_price_truth_coverage_publishes_windowed_fetch_plan(self) -> None:
        truth = replay._price_truth_coverage(
            load_stats={"raw_directional_rows": 40, "deduped_directional_rows": 40},
            candle_stats={"history_files": 1, "history_candles": 500},
            score_stats={
                "evaluated_rows": 35,
                "missing_price_window_groups": [
                    {
                        "date": "2026-06-18",
                        "count": 5,
                        "needed_from_utc": "2026-06-18T00:00:00Z",
                        "needed_to_utc": "2026-06-18T02:00:00Z",
                        "pairs": ["GBP_USD"],
                    }
                ],
            },
            sample_coverage={
                "pairs": [
                    {
                        "pair": "GBP_USD",
                        "forecast_samples": 5,
                        "evaluated_samples": 0,
                        "missing_price_truth_samples": 5,
                    }
                ],
                "under_sampled_pair_directions": [],
            },
            granularity="S5",
            edge_min_samples=30,
        )

        self.assertEqual(truth["history_fetch_command_mode"], "WINDOWED")
        self.assertEqual(truth["history_fetch_command_count"], 1)
        self.assertEqual(truth["history_fetch_commands"][0]["date"], "2026-06-18")
        self.assertIn("--pairs GBP_USD", truth["history_fetch_commands"][0]["command"])

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

    def test_load_candles_reads_compressed_history_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "history"
            pair_dir = history / "EUR_USD"
            pair_dir.mkdir(parents=True)
            path = pair_dir / "EUR_USD_S5_BA_20260619T000000Z_20260619T010000Z.jsonl.gz"
            row = {
                "pair": "EUR_USD",
                "granularity": "S5",
                "time": "2026-06-19T00:00:00Z",
                "bid": {"o": "1.1000", "h": "1.1001", "l": "1.0999", "c": "1.1000"},
                "ask": {"o": "1.1002", "h": "1.1003", "l": "1.1001", "c": "1.1002"},
            }
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

            candles, stats = replay._load_candles([history], granularity="S5")

        self.assertEqual(stats["history_files"], 1)
        self.assertEqual(stats["history_raw_rows"], 1)
        self.assertIn("EUR_USD", candles)
        self.assertEqual(len(candles["EUR_USD"]), 1)
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
