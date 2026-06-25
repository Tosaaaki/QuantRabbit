from __future__ import annotations

import importlib.util
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "oanda_universal_rotation_miner.py"
    spec = importlib.util.spec_from_file_location("oanda_universal_rotation_miner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


miner = _load_module()


class OandaUniversalRotationMinerTest(unittest.TestCase):
    def test_help_runs_without_pythonpath(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        script = repo / "scripts" / "oanda_universal_rotation_miner.py"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Mine cross-pair high-turnover entry shapes", result.stdout)

    def test_parse_exit_shapes_keeps_valid_shapes(self) -> None:
        parsed = miner._parse_exit_shapes("tp0.75_sl0.5,bad,tp1_sl1,tp0_sl1")

        self.assertEqual(parsed, (("tp0.75_sl0.5", 0.75, 0.5), ("tp1_sl1", 1.0, 1.0)))

    def test_parse_exit_shapes_rejects_empty_valid_set(self) -> None:
        with self.assertRaises(ValueError):
            miner._parse_exit_shapes("bad,tp0_sl1,tp1_sl0")

    def test_parse_multi_confluence_sizes_dedupes_and_requires_three_plus(self) -> None:
        self.assertEqual(miner._parse_multi_confluence_sizes("4,3,4"), (3, 4))
        with self.assertRaises(ValueError):
            miner._parse_multi_confluence_sizes("2")

    def test_parse_inversion_selector_sizes_allows_two_plus(self) -> None:
        self.assertEqual(miner._parse_inversion_selector_sizes("3,2,3"), (2, 3))
        with self.assertRaises(ValueError):
            miner._parse_inversion_selector_sizes("1")

    def test_filter_score_rows_keeps_requested_shape_and_side(self) -> None:
        rows = [
            {"shape": "range_reversion", "side": "LONG"},
            {"shape": "trend_continuation", "side": "SHORT"},
            {"shape": "range_reversion", "side": "SHORT"},
        ]

        filtered = miner._filter_score_rows(
            rows,
            shape_filter=("range_reversion",),
            side_filter=("SHORT",),
        )

        self.assertEqual(filtered, [{"shape": "range_reversion", "side": "SHORT"}])

    def test_discovers_and_reads_compressed_m5_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_dir = root / "20260622T155928Z" / "EUR_USD"
            pair_dir.mkdir(parents=True)
            path = pair_dir / "EUR_USD_M5_BA_20260316T000000Z_20260620T000000Z.jsonl.gz"
            row = {
                "time": "2026-06-19T00:00:00Z",
                "complete": True,
                "volume": 1,
                "bid": {"o": "1.1000", "h": "1.1001", "l": "1.0999", "c": "1.1000"},
                "ask": {"o": "1.1002", "h": "1.1003", "l": "1.1001", "c": "1.1002"},
            }
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

            files = miner._discover_m5_files(
                root,
                "EUR_USD_M5_BA_20260316T000000Z_20260620T000000Z.jsonl",
                pairs={"EUR_USD"},
            )
            candles = miner._load_ba_candles(files[0])

        self.assertEqual(files, [path])
        self.assertEqual(len(candles), 1)
        self.assertAlmostEqual(candles[0].ask_c, 1.1002)

    def test_default_history_glob_matches_rolling_oanda_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_dir = root / "20260623T041751Z" / "GBP_JPY"
            pair_dir.mkdir(parents=True)
            path = pair_dir / "GBP_JPY_M5_BA_20260223T041751Z_20260623T041751Z.jsonl"
            path.write_text("", encoding="utf-8")

            files = miner._discover_m5_files(
                root,
                miner.DEFAULT_HISTORY_GLOB,
                pairs={"GBP_JPY"},
            )

        self.assertEqual(files, [path])

    def test_select_history_files_applies_deterministic_pair_shards(self) -> None:
        files = [
            Path("/tmp/root/run/AUD_USD/AUD_USD_M5_BA.jsonl"),
            Path("/tmp/root/run/EUR_JPY/EUR_JPY_M5_BA.jsonl"),
            Path("/tmp/root/run/GBP_JPY/GBP_JPY_M5_BA.jsonl"),
            Path("/tmp/root/run/USD_JPY/USD_JPY_M5_BA.jsonl"),
        ]

        first = miner._select_history_files(files, pair_shards=2, pair_shard_index=0)
        second = miner._select_history_files(files, pair_shards=2, pair_shard_index=1)
        capped = miner._select_history_files(
            files,
            pair_shards=1,
            pair_shard_index=0,
            max_history_pairs=2,
        )

        self.assertEqual([path.parent.name for path in first], ["AUD_USD", "GBP_JPY"])
        self.assertEqual([path.parent.name for path in second], ["EUR_JPY", "USD_JPY"])
        self.assertEqual([path.parent.name for path in capped], ["AUD_USD", "EUR_JPY"])

    def test_select_history_files_rejects_invalid_resource_args(self) -> None:
        files = [Path("/tmp/root/run/USD_JPY/USD_JPY_M5_BA.jsonl")]

        with self.assertRaises(ValueError):
            miner._select_history_files(files, pair_shards=0, pair_shard_index=0)
        with self.assertRaises(ValueError):
            miner._select_history_files(files, pair_shards=2, pair_shard_index=2)
        with self.assertRaises(ValueError):
            miner._select_history_files(files, pair_shards=1, pair_shard_index=0, max_history_pairs=-1)

    def test_history_selection_stats_marks_partial_pair_scan(self) -> None:
        discovered = [
            Path("/tmp/root/run/AUD_USD/AUD_USD_M5_BA.jsonl"),
            Path("/tmp/root/run/EUR_JPY/EUR_JPY_M5_BA.jsonl"),
            Path("/tmp/root/run/GBP_JPY/GBP_JPY_M5_BA.jsonl"),
        ]
        selected = discovered[:1]

        stats = miner._history_selection_stats(
            discovered,
            selected,
            pair_shards=3,
            pair_shard_index=0,
            max_history_pairs=0,
        )

        self.assertEqual(stats["history_files_discovered"], 3)
        self.assertEqual(stats["history_pairs_discovered"], 3)
        self.assertEqual(
            stats["history_pairs_discovered_order"],
            ["AUD_USD", "EUR_JPY", "GBP_JPY"],
        )
        self.assertEqual(
            stats["history_pair_selection"]["selected_pairs"],
            ["AUD_USD"],
        )
        self.assertTrue(stats["history_pair_selection"]["is_partial_pair_scan"])

    def test_score_exit_uses_spread_floor_for_take_profit_and_stop(self) -> None:
        start = datetime(2026, 6, 1, tzinfo=timezone.utc)
        candles = [
            miner.BaOhlc(
                timestamp_utc=start,
                bid_o=1.1000,
                bid_h=1.1002,
                bid_l=1.0998,
                bid_c=1.1000,
                ask_o=1.1002,
                ask_h=1.1004,
                ask_l=1.1000,
                ask_c=1.1002,
                volume=1.0,
            ),
            miner.BaOhlc(
                timestamp_utc=start + timedelta(minutes=5),
                bid_o=1.1001,
                bid_h=1.10071,
                bid_l=1.1000,
                bid_c=1.1006,
                ask_o=1.1003,
                ask_h=1.10091,
                ask_l=1.1002,
                ask_c=1.1008,
                volume=1.0,
            ),
        ]
        candidate = miner.Candidate(
            timestamp_utc=start,
            pair="EUR_USD",
            side="LONG",
            shape="pullback_continuation",
            features=(),
            entry_bid=1.1000,
            entry_ask=1.1002,
            atr_pips=1.0,
            spread_pips=2.0,
        )

        result = miner._score_exit(
            candidate,
            candles,
            0,
            factor=10000,
            tp_atr=0.5,
            sl_atr=0.5,
            max_hold_bars=1,
            tp_spread_floor=2.5,
            sl_spread_floor=2.0,
        )

        self.assertEqual(result["outcome"], "TAKE_PROFIT_FIRST")
        self.assertAlmostEqual(result["take_profit_pips"], 5.0)
        self.assertAlmostEqual(result["stop_loss_pips"], 4.0)
        self.assertAlmostEqual(result["realized_pips"], 5.0)

    def test_direction_selector_uses_train_side_not_validation_hindsight(self) -> None:
        start = datetime(2026, 6, 1, tzinfo=timezone.utc)
        rows = []
        for idx in range(20):
            ts = start + timedelta(minutes=idx)
            in_train = idx < 14
            for side in ("LONG", "SHORT"):
                if in_train:
                    realized_atr = 0.2 if side == "LONG" else -0.1
                else:
                    realized_atr = -0.2 if side == "LONG" else 0.2
                rows.append(
                    {
                        "timestamp_utc": miner._iso(ts),
                        "jst_day": ts.astimezone(miner.JST).date().isoformat(),
                        "side": side,
                        "realized_pips": realized_atr * 10.0,
                        "realized_atr": realized_atr,
                        "win": realized_atr > 0.0,
                        "outcome": "TAKE_PROFIT_FIRST" if realized_atr > 0.0 else "STOP_FIRST",
                    }
                )

        summary = miner._train_select_side_summary(
            rows,
            train_fraction=0.7,
            min_active_days=1,
            max_daily_sample_share=1.0,
            min_positive_day_rate=0.0,
            min_validation_expectancy_atr=0.0,
            min_validation_win_rate=0.5,
            min_validation_samples=3,
            min_profit_factor=1.0,
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["selected_side"], "LONG")
        self.assertEqual(summary["qualification"], "FAIL")
        self.assertEqual(summary["validation_win_rate"], 0.0)
        self.assertIn("VALIDATION_WIN_RATE_TOO_LOW", summary["blockers"])

    def test_pair_confluence_feature_groups_ignore_identity_features(self) -> None:
        features = (
            "shape:range_reversion",
            "side:LONG",
            "session:asia",
            "session:asia",
            "atr_regime:mid",
            "spread_regime:low",
            "range_pos:low",
            "side_range:reward_edge",
        )

        groups = miner._pair_confluence_feature_groups(features, 3)

        self.assertIn(("atr_regime:mid", "range_pos:low", "session:asia"), groups)
        self.assertNotIn(("shape:range_reversion", "side:LONG", "session:asia"), groups)
        self.assertEqual(groups, miner._pair_confluence_feature_groups(features, 3))

    def test_bucket_summary_keeps_chronological_split_and_stability_metrics(self) -> None:
        start = datetime(2026, 6, 1, tzinfo=timezone.utc)
        values = []
        for index, (pair, realized_atr) in enumerate(
            (
                ("EUR_JPY", 0.2),
                ("USD_JPY", -0.1),
                ("EUR_JPY", 0.4),
                ("USD_JPY", 0.3),
                ("EUR_JPY", -0.2),
                ("USD_JPY", 0.6),
            )
        ):
            ts = start + timedelta(days=index)
            values.append(
                {
                    "timestamp_utc": miner._iso(ts),
                    "jst_day": ts.date().isoformat(),
                    "pair": pair,
                    "realized_pips": realized_atr * 10.0,
                    "realized_atr": realized_atr,
                    "win": realized_atr > 0.0,
                    "outcome": "TAKE_PROFIT_FIRST" if realized_atr > 0.0 else "STOP_FIRST",
                }
            )

        summary = miner._bucket_summary(
            list(reversed(values)),
            train_fraction=0.5,
            min_active_days=1,
            min_pair_count=2,
            max_pair_sample_share=1.0,
            max_daily_sample_share=1.0,
            min_positive_day_rate=0.0,
            min_validation_expectancy_atr=0.0,
            min_validation_win_rate=0.0,
            min_validation_samples=1,
            min_profit_factor=0.0,
        )
        chronological_summary = miner._bucket_summary(
            values,
            train_fraction=0.5,
            min_active_days=1,
            min_pair_count=2,
            max_pair_sample_share=1.0,
            max_daily_sample_share=1.0,
            min_positive_day_rate=0.0,
            min_validation_expectancy_atr=0.0,
            min_validation_win_rate=0.0,
            min_validation_samples=1,
            min_profit_factor=0.0,
            values_are_chronological=True,
        )

        self.assertEqual(summary, chronological_summary)
        self.assertEqual(summary["train_n"], 3)
        self.assertEqual(summary["validation_n"], 3)
        self.assertEqual(summary["pair_count"], 2)
        self.assertEqual(summary["active_days"], 3)
        self.assertEqual(summary["validation_avg_realized_atr"], round((0.3 - 0.2 + 0.6) / 3, 6))
        self.assertEqual(summary["qualification"], "PASS")

    def test_build_report_mines_three_and_four_feature_confluences(self) -> None:
        rows = []
        start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        features = [
            "shape:range_reversion",
            "side:LONG",
            "session:london_ny_overlap",
            "atr_regime:mid",
            "spread_regime:mid",
            "range_pos:low",
            "body:flat",
            "wick_reject:1",
            "bar_range:normal",
            "failed_break:0",
        ]
        for index in range(80):
            ts = start + timedelta(days=index)
            rows.append(
                {
                    "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
                    "jst_day": ts.date().isoformat(),
                    "pair": "USD_JPY",
                    "shape": "range_reversion",
                    "side": "LONG",
                    "exit_shape": "tp1_sl1",
                    "realized_pips": 4.0,
                    "realized_atr": 0.5,
                    "win": True,
                    "outcome": "TAKE_PROFIT_FIRST",
                    "features": features,
                    "neutral_features": [],
                }
            )

        report = miner._build_report(
            rows,
            generated_at_utc=start,
            history_root=miner.DEFAULT_HISTORY_ROOT,
            files=[],
            exit_shapes=miner._parse_exit_shapes("tp1_sl1"),
            max_hold_bars=12,
            stride_bars=1,
            tp_spread_floor=2.5,
            sl_spread_floor=2.0,
            train_fraction=0.7,
            min_samples=4,
            min_active_days=1,
            min_pair_count=1,
            max_pair_sample_share=1.0,
            max_daily_sample_share=1.0,
            min_positive_day_rate=0.0,
            min_validation_expectancy_atr=0.0,
            min_validation_win_rate=0.0,
            min_validation_samples=2,
            min_profit_factor=0.0,
            high_precision_min_win_rate=0.7,
            high_precision_min_wilson_lower=0.5,
            multi_confluence_sizes=(3, 4),
            top=100,
            load_stats={"history_files": 0, "history_pairs": 0, "scored_outcomes": len(rows)},
        )

        sizes = {
            row["confluence_size"]
            for row in report["high_precision_multi_confluences"]
        }
        self.assertEqual(report["config"]["multi_confluence_sizes"], [3, 4])
        self.assertIn(3, sizes)
        self.assertIn(4, sizes)

    def test_build_report_surfaces_live_grade_evidence_queue_for_sample_gap(self) -> None:
        rows = []
        start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        features = [
            "shape:pullback_continuation",
            "side:SHORT",
            "session:london_ny_overlap",
            "spread_regime:mid",
            "wick_reject:1",
            "bar_range:normal",
        ]
        for index in range(36):
            ts = start + timedelta(days=index)
            rows.append(
                {
                    "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
                    "jst_day": ts.date().isoformat(),
                    "pair": "EUR_USD",
                    "shape": "pullback_continuation",
                    "side": "SHORT",
                    "exit_shape": "tp1.25_sl1",
                    "realized_pips": 6.0,
                    "realized_atr": 0.8,
                    "win": True,
                    "outcome": "TAKE_PROFIT_FIRST",
                    "features": features,
                    "neutral_features": [],
                }
            )

        report = miner._build_report(
            rows,
            generated_at_utc=start,
            history_root=miner.DEFAULT_HISTORY_ROOT,
            files=[],
            exit_shapes=miner._parse_exit_shapes("tp1.25_sl1"),
            max_hold_bars=12,
            stride_bars=1,
            tp_spread_floor=2.5,
            sl_spread_floor=2.0,
            train_fraction=0.7,
            min_samples=24,
            min_active_days=15,
            min_pair_count=1,
            max_pair_sample_share=1.0,
            max_daily_sample_share=1.0,
            min_positive_day_rate=0.90,
            min_validation_expectancy_atr=0.0,
            min_validation_win_rate=0.90,
            min_validation_samples=30,
            min_profit_factor=0.0,
            high_precision_min_win_rate=0.90,
            high_precision_min_wilson_lower=0.90,
            multi_confluence_sizes=(3,),
            top=100,
            load_stats={"history_files": 0, "history_pairs": 0, "scored_outcomes": len(rows)},
        )

        self.assertEqual(report["qualified_multi_confluence_count"], 0)
        self.assertGreater(report["live_grade_evidence_queue_count"], 0)
        queue_row = next(
            row
            for row in report["live_grade_evidence_queue"]
            if row["evidence_section"] == "multi_confluence"
        )
        self.assertEqual(queue_row["pair"], "EUR_USD")
        self.assertFalse(queue_row["live_permission"])
        self.assertIn("NEEDS_MORE_VALIDATION_SAMPLES", queue_row["live_grade_gap_reasons"])
        self.assertIn("NEEDS_WILSON95_LOWER_CONFIRMATION", queue_row["live_grade_gap_reasons"])
        self.assertEqual(queue_row["missing_validation_samples"], 19)
        self.assertGreater(queue_row["additional_all_win_samples_for_wilson95_lower"], 0)
        firepower = report["campaign_firepower"]
        self.assertEqual(firepower["status"], "EVIDENCE_QUEUE_ONLY_NO_VERIFIED_FIREPOWER")
        self.assertEqual(firepower["high_precision"]["unique_vehicle_count"], 0)
        self.assertEqual(firepower["evidence_queue"]["unique_vehicle_count"], 1)
        evidence_vehicle = firepower["evidence_queue"]["top_vehicles"][0]
        self.assertFalse(evidence_vehicle["live_permission"])
        self.assertEqual(evidence_vehicle["evidence_status"], "EVIDENCE_COLLECTION_ONLY")
        self.assertEqual(evidence_vehicle["trades_needed_for_minimum_5pct"], 7)

    def test_build_report_estimates_campaign_firepower_from_verified_unique_vehicles(self) -> None:
        rows = []
        start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        pairs = ("EUR_USD", "GBP_USD", "AUD_JPY", "GBP_JPY", "USD_CAD")
        features = [
            "shape:pullback_continuation",
            "side:LONG",
            "session:london_ny_overlap",
            "atr_regime:mid",
            "spread_regime:mid",
            "range_pos:mid",
            "bar_range:normal",
        ]
        for pair in pairs:
            for index in range(80):
                ts = start + timedelta(days=index)
                rows.append(
                    {
                        "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
                        "jst_day": ts.date().isoformat(),
                        "pair": pair,
                        "shape": "pullback_continuation",
                        "side": "LONG",
                        "exit_shape": "tp1_sl1",
                        "realized_pips": 10.0,
                        "realized_atr": 1.0,
                        "win": True,
                        "outcome": "TAKE_PROFIT_FIRST",
                        "features": features,
                        "neutral_features": [],
                    }
                )

        report = miner._build_report(
            rows,
            generated_at_utc=start,
            history_root=miner.DEFAULT_HISTORY_ROOT,
            files=[],
            exit_shapes=miner._parse_exit_shapes("tp1_sl1"),
            max_hold_bars=12,
            stride_bars=1,
            tp_spread_floor=2.5,
            sl_spread_floor=2.0,
            train_fraction=0.7,
            min_samples=4,
            min_active_days=1,
            min_pair_count=1,
            max_pair_sample_share=1.0,
            max_daily_sample_share=1.0,
            min_positive_day_rate=0.0,
            min_validation_expectancy_atr=0.0,
            min_validation_win_rate=0.70,
            min_validation_samples=10,
            min_profit_factor=0.0,
            high_precision_min_win_rate=0.70,
            high_precision_min_wilson_lower=0.50,
            multi_confluence_sizes=(3,),
            top=100,
            load_stats={"history_files": 0, "history_pairs": 0, "scored_outcomes": len(rows)},
        )

        firepower = report["campaign_firepower"]
        self.assertEqual(firepower["status"], "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED")
        self.assertEqual(firepower["high_precision"]["unique_vehicle_count"], len(pairs))
        self.assertAlmostEqual(
            firepower["high_precision"][
                "estimated_return_pct_per_active_day_at_observed_frequency"
            ],
            5.0,
        )
        self.assertEqual(
            firepower["high_precision"]["trades_needed_for_minimum_5pct_at_weighted_expectancy"],
            5,
        )
        vehicle = firepower["high_precision"]["top_vehicles"][0]
        self.assertEqual(vehicle["evidence_status"], "HIGH_PRECISION_VALIDATED")
        self.assertFalse(vehicle["live_permission"])

    def test_build_report_mines_same_candle_inversion_selectors(self) -> None:
        inversion_rows = []
        start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        for index in range(80):
            ts = start + timedelta(days=index)
            inversion_rows.append(
                {
                    "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
                    "jst_day": ts.date().isoformat(),
                    "pair": "AUD_JPY",
                    "shape": "range_reversion",
                    "source_shape": "range_reversion",
                    "source_side": "LONG",
                    "selected_side": "SHORT",
                    "side": "SHORT",
                    "exit_shape": "tp1_sl1",
                    "realized_pips": 4.0,
                    "realized_atr": 0.4,
                    "win": True,
                    "outcome": "TAKE_PROFIT_FIRST",
                    "source_realized_pips": -3.0,
                    "source_realized_atr": -0.3,
                    "source_win": False,
                    "source_outcome": "STOP_FIRST",
                    "neutral_features": [
                        "session:asia",
                        "atr_regime:high",
                        "spread_regime:low",
                    ],
                }
            )

        report = miner._build_report(
            [],
            inversion_rows=inversion_rows,
            generated_at_utc=start,
            history_root=miner.DEFAULT_HISTORY_ROOT,
            files=[],
            exit_shapes=miner._parse_exit_shapes("tp1_sl1"),
            max_hold_bars=12,
            stride_bars=1,
            tp_spread_floor=2.5,
            sl_spread_floor=2.0,
            train_fraction=0.7,
            min_samples=4,
            min_active_days=1,
            min_pair_count=1,
            max_pair_sample_share=1.0,
            max_daily_sample_share=1.0,
            min_positive_day_rate=0.0,
            min_validation_expectancy_atr=0.0,
            min_validation_win_rate=0.0,
            min_validation_samples=2,
            min_profit_factor=0.0,
            high_precision_min_win_rate=0.7,
            high_precision_min_wilson_lower=0.5,
            multi_confluence_sizes=(3,),
            inversion_selector_sizes=(2,),
            top=100,
            load_stats={"history_files": 0, "history_pairs": 0, "scored_outcomes": 0},
        )

        self.assertGreater(report["qualified_inversion_selector_count"], 0)
        self.assertGreater(report["high_precision_inversion_selector_count"], 0)
        row = report["high_precision_inversion_selectors"][0]
        self.assertEqual(row["pair"], "AUD_JPY")
        self.assertEqual(row["source_side"], "LONG")
        self.assertEqual(row["selected_side"], "SHORT")
        self.assertLess(row["source_validation_avg_realized_atr"], 0.0)
        self.assertGreater(row["validation_inversion_edge_atr"], 0.0)

    def test_inversion_selector_requires_source_side_to_fail(self) -> None:
        start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        rows = []
        for index in range(80):
            ts = start + timedelta(days=index)
            rows.append(
                {
                    "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
                    "jst_day": ts.date().isoformat(),
                    "pair": "AUD_JPY",
                    "shape": "range_reversion",
                    "source_shape": "range_reversion",
                    "source_side": "LONG",
                    "selected_side": "SHORT",
                    "side": "SHORT",
                    "exit_shape": "tp1_sl1",
                    "realized_pips": 4.0,
                    "realized_atr": 0.4,
                    "win": True,
                    "outcome": "TAKE_PROFIT_FIRST",
                    "source_realized_pips": 1.0,
                    "source_realized_atr": 0.1,
                    "source_win": True,
                    "source_outcome": "TAKE_PROFIT_FIRST",
                    "neutral_features": [
                        "session:asia",
                        "atr_regime:high",
                    ],
                }
            )

        report = miner._build_report(
            [],
            inversion_rows=rows,
            generated_at_utc=start,
            history_root=miner.DEFAULT_HISTORY_ROOT,
            files=[],
            exit_shapes=miner._parse_exit_shapes("tp1_sl1"),
            max_hold_bars=12,
            stride_bars=1,
            tp_spread_floor=2.5,
            sl_spread_floor=2.0,
            train_fraction=0.7,
            min_samples=4,
            min_active_days=1,
            min_pair_count=1,
            max_pair_sample_share=1.0,
            max_daily_sample_share=1.0,
            min_positive_day_rate=0.0,
            min_validation_expectancy_atr=0.0,
            min_validation_win_rate=0.0,
            min_validation_samples=2,
            min_profit_factor=0.0,
            high_precision_min_win_rate=0.7,
            high_precision_min_wilson_lower=0.5,
            multi_confluence_sizes=(3,),
            inversion_selector_sizes=(2,),
            top=100,
            load_stats={"history_files": 0, "history_pairs": 0, "scored_outcomes": 0},
        )

        self.assertEqual(report["qualified_inversion_selector_count"], 0)
        self.assertIn("TRAIN_SOURCE_NOT_NEGATIVE", report["top_inversion_selectors"][0]["blockers"])
        self.assertIn("VALIDATION_SOURCE_NOT_NEGATIVE", report["top_inversion_selectors"][0]["blockers"])


if __name__ == "__main__":
    unittest.main()
