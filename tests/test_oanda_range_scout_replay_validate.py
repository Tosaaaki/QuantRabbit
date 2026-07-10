from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = _load_script("oanda_history_replay_validate")
range_replay = _load_script("oanda_range_scout_replay_validate")


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _row(*, current: float = 1.105, low: float = 1.100, high: float = 1.110):
    return range_replay.RangeForecastRow(
        source_index=1,
        timestamp_utc=_dt("2026-07-01T00:00:00Z"),
        pair="EUR_USD",
        confidence=0.8,
        current_price=current,
        range_low_price=low,
        range_high_price=high,
        horizon_min=120.0,
        cycle_id="cycle-1",
    )


def _candle(
    *,
    bid_o: float,
    bid_h: float,
    bid_l: float,
    bid_c: float,
    ask_o: float,
    ask_h: float,
    ask_l: float,
    ask_c: float,
    timestamp: str = "2026-07-01T00:00:05Z",
):
    return replay.QuoteCandle(
        timestamp_utc=_dt(timestamp),
        pair="EUR_USD",
        bid=replay.Ohlc(bid_o, bid_h, bid_l, bid_c),
        ask=replay.Ohlc(ask_o, ask_h, ask_l, ask_c),
    )


class OandaRangeScoutReplayValidateTest(unittest.TestCase):
    def test_loader_dedupes_cycle_and_overlapping_persistence(self) -> None:
        rows = [
            {
                "timestamp_utc": "2026-07-01T00:00:00Z",
                "cycle_id": "cycle-a",
                "pair": "EUR_USD",
                "direction": "RANGE",
                "confidence": 0.8,
                "current_price": 1.105,
                "range_low_price": 1.100,
                "range_high_price": 1.110,
                "horizon_min": 120,
            },
            {
                "timestamp_utc": "2026-07-01T00:00:10Z",
                "cycle_id": "cycle-a",
                "pair": "EUR_USD",
                "direction": "RANGE",
                "confidence": 0.8,
                "current_price": 1.105,
                "range_low_price": 1.100,
                "range_high_price": 1.110,
                "horizon_min": 120,
            },
            {
                "timestamp_utc": "2026-07-01T00:30:00Z",
                "cycle_id": "cycle-b",
                "pair": "EUR_USD",
                "direction": "RANGE",
                "confidence": 0.8,
                "current_price": 1.105,
                "range_low_price": 1.100,
                "range_high_price": 1.110,
                "horizon_min": 120,
            },
            {
                "timestamp_utc": "2026-07-01T01:40:00Z",
                "cycle_id": "cycle-c",
                "pair": "EUR_USD",
                "direction": "RANGE",
                "confidence": 0.8,
                "current_price": 1.106,
                "range_low_price": 1.101,
                "range_high_price": 1.111,
                "horizon_min": 120,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forecast.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in rows) + "\n")
            loaded, stats = range_replay.load_range_forecasts(path, dedupe_minutes=90)

        self.assertEqual([item.cycle_id for item in loaded], ["cycle-a", "cycle-c"])
        self.assertEqual(stats["skipped_duplicate_rows"], 1)
        self.assertEqual(stats["skipped_persistence_rows"], 1)

    def test_nearest_rail_selects_one_side(self) -> None:
        lower = range_replay.range_signal(_row(current=1.103))
        upper = range_replay.range_signal(_row(current=1.108))

        self.assertEqual((lower.side, lower.entry), ("LONG", 1.100))
        self.assertEqual((upper.side, upper.entry), ("SHORT", 1.110))

    def test_long_same_candle_target_and_stop_is_stop_first(self) -> None:
        signal = range_replay.range_signal(_row(current=1.103))
        candle = _candle(
            bid_o=1.1000,
            bid_h=1.1012,
            bid_l=1.0994,
            bid_c=1.1005,
            ask_o=1.1002,
            ask_h=1.1014,
            ask_l=1.0996,
            ask_c=1.1007,
        )

        result = range_replay.simulate_filled_signal(
            signal,
            [candle],
            take_profit_pips=10.0,
            stop_loss_pips=5.0,
            pip=0.0001,
        )

        self.assertEqual(result["exit_reason"], "STOP_LOSS")
        self.assertEqual(result["realized_pips"], -5.0)
        self.assertTrue(result["same_candle_stop_first"])

    def test_fill_candle_target_only_uses_adverse_ordering_without_deleting_sample(self) -> None:
        signal = range_replay.range_signal(_row(current=1.103))
        candle = _candle(
            bid_o=1.1000,
            bid_h=1.1012,
            bid_l=1.0995,
            bid_c=1.1004,
            ask_o=1.1002,
            ask_h=1.1014,
            ask_l=1.0997,
            ask_c=1.1006,
        )

        result = range_replay.simulate_filled_signal(
            signal,
            [candle],
            take_profit_pips=10.0,
            stop_loss_pips=10.0,
            pip=0.0001,
        )

        self.assertEqual(result["exit_reason"], "TTL_CLOSE")
        self.assertAlmostEqual(result["realized_pips"], 4.0)
        self.assertTrue(result["scorable"])
        self.assertTrue(result["fill_bar_target_ambiguous"])

    def test_fill_candle_close_beyond_target_proves_post_fill_take_profit(self) -> None:
        signal = range_replay.range_signal(_row(current=1.103))
        candle = _candle(
            bid_o=1.1000,
            bid_h=1.1012,
            bid_l=1.0995,
            bid_c=1.1011,
            ask_o=1.1002,
            ask_h=1.1014,
            ask_l=1.0997,
            ask_c=1.1013,
        )

        result = range_replay.simulate_filled_signal(
            signal,
            [candle],
            take_profit_pips=10.0,
            stop_loss_pips=10.0,
            pip=0.0001,
        )

        self.assertEqual(result["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(result["realized_pips"], 10.0)
        self.assertTrue(result["fill_bar_target_ambiguous"])

    def test_timeout_close_uses_executable_bid_for_long(self) -> None:
        signal = range_replay.range_signal(_row(current=1.103))
        candle = _candle(
            bid_o=1.1000,
            bid_h=1.1004,
            bid_l=1.0998,
            bid_c=1.1003,
            ask_o=1.1002,
            ask_h=1.1006,
            ask_l=1.1000,
            ask_c=1.1005,
        )

        result = range_replay.simulate_filled_signal(
            signal,
            [candle],
            take_profit_pips=10.0,
            stop_loss_pips=10.0,
            pip=0.0001,
        )

        self.assertEqual(result["exit_reason"], "TTL_CLOSE")
        self.assertAlmostEqual(result["realized_pips"], 3.0)

    def test_timeout_close_uses_executable_ask_for_short(self) -> None:
        signal = range_replay.range_signal(_row(current=1.108))
        candle = _candle(
            bid_o=1.1098,
            bid_h=1.1100,
            bid_l=1.1094,
            bid_c=1.1095,
            ask_o=1.1100,
            ask_h=1.1102,
            ask_l=1.1096,
            ask_c=1.1097,
        )

        result = range_replay.simulate_filled_signal(
            signal,
            [candle],
            take_profit_pips=10.0,
            stop_loss_pips=10.0,
            pip=0.0001,
        )

        self.assertEqual(result["exit_reason"], "TTL_CLOSE")
        self.assertAlmostEqual(result["realized_pips"], 3.0)

    def test_score_does_not_read_candle_that_starts_at_ttl_boundary(self) -> None:
        row = _row(current=1.103)
        pre_ttl_candles = [
            _candle(
                timestamp=f"2026-07-01T00:00:{second:02d}Z",
                bid_o=1.1000,
                bid_h=1.1004,
                bid_l=1.0998,
                bid_c=1.1003,
                ask_o=1.1002,
                ask_h=1.1006,
                ask_l=1.1000,
                ask_c=1.1005,
            )
            for second in range(0, 60, 5)
        ]
        after_ttl_target = _candle(
            timestamp="2026-07-01T00:01:00Z",
            bid_o=1.1003,
            bid_h=1.1012,
            bid_l=1.1002,
            bid_c=1.1011,
            ask_o=1.1005,
            ask_h=1.1014,
            ask_l=1.1004,
            ask_c=1.1013,
        )

        results, stats = range_replay.score_range_forecasts(
            [row],
            {"EUR_USD": [*pre_ttl_candles, after_ttl_target]},
            ttl_minutes=1,
            tp_grid=(10.0,),
            sl_grid=(10.0,),
            candle_interval=timedelta(seconds=5),
        )

        self.assertEqual(stats["filled_signals"], 1)
        self.assertEqual(results[0]["exit_reason"], "TTL_CLOSE")
        self.assertAlmostEqual(results[0]["realized_pips"], 3.0)

    def test_incomplete_truth_tail_is_not_scored_as_ttl_close(self) -> None:
        row = _row(current=1.103)
        lone_fill = _candle(
            bid_o=1.1000,
            bid_h=1.1004,
            bid_l=1.0998,
            bid_c=1.1003,
            ask_o=1.1002,
            ask_h=1.1006,
            ask_l=1.1000,
            ask_c=1.1005,
        )

        results, stats = range_replay.score_range_forecasts(
            [row],
            {"EUR_USD": [lone_fill]},
            ttl_minutes=90,
            tp_grid=(10.0,),
            sl_grid=(10.0,),
            candle_interval=timedelta(seconds=5),
        )

        self.assertEqual(results, [])
        self.assertEqual(stats["skipped_incomplete_truth_window"], 1)

    def test_truth_window_rejects_a_full_missing_interval_at_start(self) -> None:
        first = _candle(timestamp="2026-07-01T00:00:05Z", **{
            "bid_o": 1.1000,
            "bid_h": 1.1004,
            "bid_l": 1.0998,
            "bid_c": 1.1003,
            "ask_o": 1.1002,
            "ask_h": 1.1006,
            "ask_l": 1.1000,
            "ask_c": 1.1005,
        })

        self.assertFalse(
            range_replay.complete_truth_window(
                [first],
                forecast_start=_dt("2026-07-01T00:00:00Z"),
                forecast_end=_dt("2026-07-01T00:00:10Z"),
                candle_interval=timedelta(seconds=5),
            )
        )

    def test_friday_no_market_classification_uses_ttl_not_forecast_horizon(self) -> None:
        friday_16_ny = replace(
            _row(),
            timestamp_utc=_dt("2026-07-03T20:00:00Z"),
            horizon_min=360.0,
        )

        self.assertFalse(
            range_replay.range_truth_is_no_market(friday_16_ny, ttl_minutes=60)
        )
        self.assertTrue(
            range_replay.range_truth_is_no_market(friday_16_ny, ttl_minutes=90)
        )

    def test_nonpositive_grid_distance_is_rejected(self) -> None:
        signal = range_replay.range_signal(_row(current=1.103))
        candle = _candle(
            bid_o=1.1000,
            bid_h=1.1004,
            bid_l=1.0998,
            bid_c=1.1003,
            ask_o=1.1002,
            ask_h=1.1006,
            ask_l=1.1000,
            ask_c=1.1005,
        )

        with self.assertRaises(ValueError):
            range_replay.simulate_filled_signal(
                signal,
                [candle],
                take_profit_pips=10.0,
                stop_loss_pips=-5.0,
                pip=0.0001,
            )

    def test_oos_status_is_insufficient_when_no_family_can_be_evaluated(self) -> None:
        self.assertEqual(
            range_replay._report_status([]),
            "INSUFFICIENT_OOS_SAMPLE_UNDER_TTL_CLOSE_RESEARCH_CONTRACT",
        )

    def test_student_t_lower_bound_does_not_use_optimistic_normal_cutoff(self) -> None:
        items = []
        for idx, pnl in enumerate([2.7] * 5 + [-0.7] * 5):
            day = f"2026-07-{idx + 1:02d}"
            items.append(
                {
                    "timestamp_utc": f"{day}T00:00:00Z",
                    "resolved_day_utc": day,
                    "realized_pips": pnl,
                    "exit_reason": "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
                }
            )

        result = range_replay.metrics(items)

        self.assertGreater(result["mean_pips"], 0.0)
        self.assertLess(result["one_sided_95_mean_lower_pips"], 0.0)

    def test_single_resolved_day_validation_is_underpowered(self) -> None:
        items = []
        for idx in range(40):
            forecast_day = idx // 4 + 1
            resolved_day = (
                f"2026-07-{forecast_day:02d}"
                if forecast_day <= 7
                else "2026-07-10"
            )
            items.append(
                {
                    "source_index": idx,
                    "timestamp_utc": f"2026-07-{forecast_day:02d}T00:00:00Z",
                    "resolved_day_utc": resolved_day,
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "take_profit_pips": 5.0,
                    "stop_loss_pips": 5.0,
                    "realized_pips": 5.0,
                    "exit_reason": "TAKE_PROFIT",
                }
            )

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["validation"]["active_days"], 1)
        self.assertFalse(selected[0]["oos_powered"])
        self.assertFalse(selected[0]["hypothesis_candidate"])
        self.assertEqual(
            range_replay._report_status(selected),
            "INSUFFICIENT_OOS_SAMPLE_UNDER_TTL_CLOSE_RESEARCH_CONTRACT",
        )

    def test_validation_edge_concentrated_in_one_day_is_not_candidate(self) -> None:
        items = []
        source_index = 0
        for day in range(1, 11):
            for sample in range(3):
                items.append(
                    {
                        "source_index": source_index,
                        "timestamp_utc": f"2026-06-{day:02d}T00:00:{sample:02d}Z",
                        "resolved_day_utc": f"2026-06-{day:02d}",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit_pips": 5.0,
                        "stop_loss_pips": 5.0,
                        "realized_pips": 5.0,
                        "exit_reason": "TAKE_PROFIT",
                    }
                )
                source_index += 1
        for sample in range(16):
            items.append(
                {
                    "source_index": source_index,
                    "timestamp_utc": f"2026-06-11T00:00:{sample:02d}Z",
                    "resolved_day_utc": "2026-06-11",
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "take_profit_pips": 5.0,
                    "stop_loss_pips": 5.0,
                    "realized_pips": 5.0,
                    "exit_reason": "TAKE_PROFIT",
                }
            )
            source_index += 1
        for day in range(12, 16):
            items.append(
                {
                    "source_index": source_index,
                    "timestamp_utc": f"2026-06-{day:02d}T00:00:00Z",
                    "resolved_day_utc": f"2026-06-{day:02d}",
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "take_profit_pips": 5.0,
                    "stop_loss_pips": 5.0,
                    "realized_pips": -5.0,
                    "exit_reason": "STOP_LOSS",
                }
            )
            source_index += 1

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["validation"]["positive_day_rate"], 0.2)
        self.assertEqual(selected[0]["validation"]["max_daily_sample_share"], 0.8)
        self.assertFalse(selected[0]["hypothesis_candidate"])

    def test_low_pair_side_truth_coverage_is_not_powered_or_candidate(self) -> None:
        items = []
        for day in range(1, 16):
            for sample in range(3):
                items.append(
                    {
                        "source_index": len(items),
                        "timestamp_utc": f"2026-06-{day:02d}T00:00:{sample:02d}Z",
                        "resolved_day_utc": f"2026-06-{day:02d}",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit_pips": 5.0,
                        "stop_loss_pips": 5.0,
                        "realized_pips": 5.0,
                        "exit_reason": "TAKE_PROFIT",
                    }
                )

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
            family_truth_coverage={
                ("EUR_USD", "LONG"): {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "forecast_rows": 450,
                    "complete_truth_windows": 45,
                    "coverage_rate": 0.1,
                }
            },
        )

        self.assertEqual(len(selected), 1)
        self.assertFalse(selected[0]["oos_powered"])
        self.assertFalse(selected[0]["hypothesis_candidate"])
        self.assertEqual(
            range_replay._report_status(selected),
            "INSUFFICIENT_OOS_SAMPLE_UNDER_TTL_CLOSE_RESEARCH_CONTRACT",
        )

    def test_validation_slice_truth_coverage_cannot_hide_behind_full_period(self) -> None:
        items = []
        for day in range(1, 16):
            for sample in range(3):
                items.append(
                    {
                        "source_index": len(items),
                        "timestamp_utc": f"2026-06-{day:02d}T00:00:{sample:02d}Z",
                        "resolved_day_utc": f"2026-06-{day:02d}",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit_pips": 5.0,
                        "stop_loss_pips": 5.0,
                        "realized_pips": 5.0,
                        "exit_reason": "TAKE_PROFIT",
                    }
                )
        forecast_by_day = {
            f"2026-06-{day:02d}": 100 if day <= 10 else 10
            for day in range(1, 16)
        }
        complete_by_day = {
            f"2026-06-{day:02d}": 100 if day <= 10 else 3
            for day in range(1, 16)
        }

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
            family_truth_coverage={
                ("EUR_USD", "LONG"): {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "forecast_rows": 1050,
                    "complete_truth_windows": 1015,
                    "coverage_rate": round(1015 / 1050, 6),
                    "forecast_rows_by_day": forecast_by_day,
                    "complete_truth_windows_by_day": complete_by_day,
                }
            },
        )

        self.assertEqual(len(selected), 1)
        coverage = selected[0]["truth_coverage"]
        self.assertGreaterEqual(
            coverage["coverage_rate"],
            range_replay.MIN_TRUTH_WINDOW_COVERAGE_RATE,
        )
        self.assertEqual(coverage["train"]["coverage_rate"], 1.0)
        self.assertEqual(coverage["validation"]["coverage_rate"], 0.3)
        self.assertFalse(selected[0]["oos_powered"])
        self.assertFalse(selected[0]["hypothesis_candidate"])

    def test_trade_level_edge_with_negative_daily_t_lower_bound_is_not_candidate(self) -> None:
        positive_days = {1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 15, 16, 17, 18}
        items = []
        for day in range(1, 21):
            pnls = [3.0] * 7 if day in positive_days else [-15.0]
            for slot, pnl in enumerate(pnls):
                items.append(
                    {
                        "source_index": len(items),
                        "timestamp_utc": (
                            f"2026-06-{day:02d}T{slot * 2:02d}:00:00Z"
                        ),
                        "resolved_day_utc": f"2026-06-{day:02d}",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit_pips": 3.0,
                        "stop_loss_pips": 15.0,
                        "realized_pips": pnl,
                        "exit_reason": "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
                    }
                )

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
        )

        self.assertEqual(len(selected), 1)
        validation = selected[0]["validation"]
        self.assertGreater(validation["one_sided_95_mean_lower_pips"], 0.0)
        self.assertLess(validation["one_sided_95_daily_mean_lower_pips"], 0.0)
        self.assertFalse(selected[0]["hypothesis_candidate"])

    def test_train_only_edge_is_not_candidate(self) -> None:
        items = []
        for idx in range(40):
            pnl = 5.0 if idx < 28 else -5.0
            items.append(
                {
                    "source_index": idx,
                    "timestamp_utc": f"2026-07-{idx // 4 + 1:02d}T00:00:00Z",
                    "campaign_day_jst": f"2026-07-{idx // 4 + 1:02d}",
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "take_profit_pips": 5.0,
                    "stop_loss_pips": 5.0,
                    "realized_pips": pnl,
                    "exit_reason": "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
                }
            )

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
        )

        self.assertEqual(len(selected), 1)
        self.assertGreater(selected[0]["train"]["mean_pips"], 0)
        self.assertLess(selected[0]["validation"]["mean_pips"], 0)
        self.assertFalse(selected[0]["hypothesis_candidate"])


if __name__ == "__main__":
    unittest.main()
