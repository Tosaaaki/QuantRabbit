from __future__ import annotations

import importlib.util
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "technical_entry_miner.py"


def _load_miner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("technical_entry_miner", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


miner = _load_miner()


def _row(
    index: int,
    *,
    features: set[str],
    hit: bool,
    final_pips: float,
) -> dict:
    timestamp = datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
    return {
        "source_index": index,
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "pair": "EUR_USD",
        "direction": "UP",
        "features": sorted({"pair:EUR_USD", "direction:UP", *features}),
        "final_direction_hit": hit,
        "final_pips": final_pips,
        "mfe_ge_2pip": hit,
        "mfe_ge_5pip": False,
        "target_touch_hit": hit,
        "target_before_invalidation_hit": hit,
        "scalp_take_profit_before_stop_hit": hit,
    }


class TechnicalEntryMinerConfluenceTest(unittest.TestCase):
    def test_confluence_candidates_require_holdout_success(self) -> None:
        good = {"M1:macd_hist_aligned", "M5:ema50_aligned"}
        overfit = {"M1:rsi_low", "M5:stoch_low"}
        rows: list[dict] = []
        for index in range(24):
            rows.append(_row(index, features={*good, *overfit}, hit=True, final_pips=1.4))
        for index in range(24, 32):
            rows.append(_row(index, features=good, hit=True, final_pips=1.1))
        for index in range(32, 48):
            rows.append(_row(index, features=overfit, hit=False, final_pips=-1.0))

        candidates = miner._mine_confluence_buckets(
            rows,
            min_train_samples=12,
            min_validation_samples=8,
            max_size=2,
            train_fraction=0.50,
            min_validation_mfe2_rate=0.75,
            feature_limit_per_row=20,
        )
        labels = [candidate["confluence"] for candidate in candidates]

        self.assertTrue(
            any(
                "M1:macd_hist_aligned" in label and "M5:ema50_aligned" in label
                for label in labels
            )
        )
        self.assertFalse(
            any("M1:rsi_low" in label and "M5:stoch_low" in label for label in labels)
        )
        best = next(
            candidate
            for candidate in candidates
            if "M1:macd_hist_aligned" in candidate["confluence"]
            and "M5:ema50_aligned" in candidate["confluence"]
        )
        self.assertEqual(best["train_n"], 24)
        self.assertEqual(best["validation_n"], 8)
        self.assertEqual(best["validation_mfe_ge_2pip_rate"], 1.0)
        self.assertGreater(best["validation_avg_final_pips"], 0.0)

    def test_confluence_features_exclude_identity_labels(self) -> None:
        features = miner._confluence_features(
            {
                "pair:EUR_USD",
                "direction:DOWN",
                "confidence:0.75-0.90",
                "M1:ema20_aligned",
                "cross:M1M5:macd_all_aligned",
            },
            limit=10,
        )

        self.assertEqual(features, ["cross:M1M5:macd_all_aligned", "M1:ema20_aligned"])


if __name__ == "__main__":
    unittest.main()
