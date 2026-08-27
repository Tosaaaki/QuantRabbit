import inspect
import hashlib
import json
import math
import unittest
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from causal_composite_indicators_v3 import Bar
from fx_original_indicators import load_bars
from run_asian_displacement_handoff_fade_v32 import (
    TRAINING_ABS_DISPLACEMENT_Q75,
    UNIVERSE,
    detect_day_signals,
)


def bar(pair, index, close, spread=0.0002):
    stamp = (datetime(2026, 3, 11, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace(
        "+00:00", ".000000000Z"
    )
    high, low = close + 0.0001, close - 0.0001
    return Bar(
        pair, stamp,
        close - spread / 2, high - spread / 2, low - spread / 2, close - spread / 2,
        close + spread / 2, high + spread / 2, low + spread / 2, close + spread / 2, 100,
    )


def fixture(signs, scale=1.1):
    result = {}
    for pair in sorted(UNIVERSE):
        target = signs[pair] * TRAINING_ABS_DISPLACEMENT_Q75[pair] * scale
        result[pair] = [
            bar(pair, index, math.exp(target * min(index, 71) / 71))
            for index in range(144)
        ]
    return result


class AsianDisplacementHandoffFadeV32Test(unittest.TestCase):
    def test_tail_displacement_emits_native_fade_for_each_pair(self):
        signs = {pair: (1 if index % 2 == 0 else -1) for index, pair in enumerate(sorted(UNIVERSE))}
        signals = detect_day_signals(fixture(signs))
        self.assertEqual(len(signals), 7)
        by_pair = {signal["pair"]: signal for signal in signals}
        for pair in UNIVERSE:
            self.assertEqual(by_pair[pair]["direction"], -signs[pair])
            self.assertEqual(
                by_pair[pair]["diagnostics"]["training_abs_displacement_q75"],
                TRAINING_ABS_DISPLACEMENT_Q75[pair],
            )

    def test_subthreshold_displacement_emits_no_signal(self):
        signs = {pair: 1 for pair in UNIVERSE}
        self.assertEqual(detect_day_signals(fixture(signs, scale=0.9)), [])

    def test_future_fill_and_exit_values_cannot_change_decision(self):
        signs = {pair: 1 for pair in UNIVERSE}
        bars = fixture(signs)
        before = detect_day_signals(bars)
        for pair in UNIVERSE:
            bars[pair][72] = bar(pair, 72, 8.0)
            bars[pair][143] = bar(pair, 143, 0.2)
        self.assertEqual(before, detect_day_signals(bars))

    def test_missing_pair_or_completed_decision_bar_fails_closed(self):
        signs = {pair: 1 for pair in UNIVERSE}
        missing_pair = fixture(signs)
        missing_pair.pop(sorted(UNIVERSE)[0])
        self.assertEqual(detect_day_signals(missing_pair), [])
        missing_bar = fixture(signs)
        del missing_bar[sorted(UNIVERSE)[0]][12]
        self.assertEqual(detect_day_signals(missing_bar), [])

    def test_signal_detector_has_no_cost_or_outcome_parameter(self):
        self.assertEqual(set(inspect.signature(detect_day_signals).parameters), {"pair_day_bars"})

    def test_thresholds_are_positive_and_cover_frozen_universe(self):
        self.assertEqual(set(TRAINING_ABS_DISPLACEMENT_Q75), set(UNIVERSE))
        self.assertTrue(all(value > 0 for value in TRAINING_ABS_DISPLACEMENT_Q75.values()))

    def test_training_thresholds_reproduce_nearest_rank_without_post_entry_outcomes(self):
        root = Path(__file__).resolve().parent
        registry = json.loads((root / "PAPER_RESEARCH_CYCLE_REGISTRY_V2.json").read_text())
        cycle = next(item for item in registry["cycles"] if item["cycle_id"] == "V32")
        source_root = Path(cycle["source_contract"]["root"])
        for pair in sorted(UNIVERSE):
            source = next((source_root / pair).glob("*_M5_BA_*.jsonl.gz"))
            by_day = defaultdict(list)
            for source_bar in load_bars(source):
                by_day[source_bar.time[:10]].append(source_bar)
            values = []
            for day, bars in sorted(by_day.items()):
                if not ("2026-03-11" <= day < "2026-05-01"):
                    continue
                by_minute = {item.time[11:16]: item for item in bars}
                if "00:00" not in by_minute or "05:55" not in by_minute:
                    continue
                start = by_minute["00:00"]
                completed = by_minute["05:55"]
                values.append(abs(math.log(completed.mid_c / start.mid_o)))
            values.sort()
            nearest_rank = values[math.ceil(0.75 * len(values)) - 1]
            self.assertEqual(nearest_rank, TRAINING_ABS_DISPLACEMENT_Q75[pair])

    def test_prereg_registry_authority_holdout_and_v32_work_order_are_frozen(self):
        root = Path(__file__).resolve().parent
        prereg = json.loads((root / "ASIAN_DISPLACEMENT_HANDOFF_FADE_PREREGISTRATION_V32.json").read_text())
        registry = json.loads((root / "PAPER_RESEARCH_CYCLE_REGISTRY_V2.json").read_text())
        cycle = next(item for item in registry["cycles"] if item["cycle_id"] == "V32")
        work_order = root / "evidence/orchestrator_state_v2/next_hypothesis_work_order_v32.json"
        self.assertEqual(
            hashlib.sha256(work_order.read_bytes()).hexdigest(),
            "82ed0e702c7691ce424ffeb75283c4a711356565f36ea59c3a454696f47b4d26",
        )
        self.assertEqual(prereg["training_only_family_selection"]["candidate_signal_families_preregistered"], 1)
        self.assertEqual(prereg["training_only_family_selection"]["candidate_signal_families_compared_by_outcome"], 0)
        self.assertFalse(prereg["frozen_execution_contract"]["changed_from_v31"])
        self.assertEqual(cycle["inventory_contract"]["finite_max_age_seconds"], 345600)
        self.assertEqual(cycle["evaluation_contract"]["holdout"]["state"], "UNOPENED")
        self.assertFalse(prereg["authority"]["live_authority"])
        self.assertFalse(prereg["authority"]["order_endpoint"])
        self.assertEqual(prereg["authority"]["external_orders"], 0)


if __name__ == "__main__":
    unittest.main()
