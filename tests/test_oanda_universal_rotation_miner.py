from __future__ import annotations

import importlib.util
import sys
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
        for index in range(40):
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


if __name__ == "__main__":
    unittest.main()
