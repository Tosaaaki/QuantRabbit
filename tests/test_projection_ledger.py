"""Unit tests for strategy/projection_ledger.py."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.strategy.projection_ledger import (
    CONFIDENCE_MAX_MULTIPLIER,
    CONFIDENCE_MIN_MULTIPLIER,
    LedgerEntry,
    compute_hit_rates,
    confidence_calibration,
    load_ledger,
    record_projections,
    setup_grade,
    verify_pending,
)


@dataclass
class _Sig:
    name: str
    direction: str
    lead_time_min: float
    confidence: float
    bonus_magnitude: float
    rationale: str = ""


class RecordTest(unittest.TestCase):
    def test_record_writes_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sigs = [
                _Sig("bb_squeeze", "EITHER", 75, 0.6, 8.0),
                _Sig("liquidity_sweep_high", "UP", 15, 0.9, 12.0, "M5 equal-highs at 1.17150 (1.0pip up)"),
            ]
            n = record_projections(sigs, pair="EUR_USD", current_price=1.17140,
                                    data_root=root)
            self.assertEqual(n, 2)
            entries = load_ledger(root)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].signal_name, "bb_squeeze")
            self.assertEqual(entries[0].resolution_status, "PENDING")
            self.assertEqual(entries[1].predicted_target_price, 1.17150)

    def test_record_empty_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            n = record_projections([], pair="EUR_USD", current_price=1.0, data_root=Path(tmp))
            self.assertEqual(n, 0)


class VerifyTest(unittest.TestCase):
    def _setup(self, signal, ts_offset_min=30):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        emitted = datetime.now(timezone.utc) - timedelta(minutes=ts_offset_min)
        entry = LedgerEntry(
            timestamp_emitted_utc=emitted.isoformat().replace("+00:00", "Z"),
            pair="EUR_USD",
            signal_name=signal.get("name", "test"),
            direction=signal.get("direction", "UP"),
            lead_time_min=signal.get("lead_time_min", 10),
            confidence=0.7,
            entry_price=signal.get("entry_price", 1.1700),
            predicted_target_price=signal.get("target"),
            resolution_window_min=signal.get("window", 20),
            resolution_status="PENDING",
        )
        from quant_rabbit.strategy.projection_ledger import write_ledger
        write_ledger([entry], root)
        return tmp, root

    def test_directional_up_hit(self) -> None:
        tmp, root = self._setup({"direction": "UP", "lead_time_min": 5, "window": 10, "entry_price": 1.1700})
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1715, "ask": 1.1716}},
                atr_pips_by_pair={"EUR_USD": 10.0},  # threshold = 5 pip
            )
            self.assertEqual(counts["HIT"], 1)
            entries = load_ledger(root)
            self.assertEqual(entries[0].resolution_status, "HIT")

    def test_directional_up_miss(self) -> None:
        tmp, root = self._setup({"direction": "UP", "lead_time_min": 5, "window": 10, "entry_price": 1.1700})
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1700, "ask": 1.1701}},  # no move
                atr_pips_by_pair={"EUR_USD": 10.0},
            )
            self.assertEqual(counts["MISS"], 1)

    def test_liquidity_target_hit(self) -> None:
        tmp, root = self._setup({
            "direction": "UP", "lead_time_min": 5, "window": 10,
            "entry_price": 1.1700, "target": 1.1720,
        })
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1719, "ask": 1.1721}},
                atr_pips_by_pair={"EUR_USD": 10.0},
            )
            # Mid 1.1720 reaches target → HIT
            self.assertEqual(counts["HIT"], 1)

    def test_pending_within_window(self) -> None:
        # Emitted just now with 60-min window → still pending
        tmp, root = self._setup({"window": 60}, ts_offset_min=10)
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.0, "ask": 1.0}},
            )
            self.assertEqual(counts["PENDING"], 1)


class HitRatesTest(unittest.TestCase):
    def test_compute_hit_rates_per_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = []
            # 3 HITs and 1 MISS for bb_squeeze on EUR_USD
            for status in ["HIT", "HIT", "HIT", "MISS"]:
                entries.append(LedgerEntry(
                    timestamp_emitted_utc="2026-05-14T00:00:00Z",
                    pair="EUR_USD", signal_name="bb_squeeze", direction="EITHER",
                    lead_time_min=10, confidence=0.7,
                    entry_price=1.0, predicted_target_price=None,
                    resolution_window_min=20, resolution_status=status,
                ))
            write_ledger(entries, root)
            hr = compute_hit_rates(root)
            self.assertIn("bb_squeeze", hr)
            self.assertEqual(hr["bb_squeeze"]["EUR_USD"]["samples"], 4)
            self.assertAlmostEqual(hr["bb_squeeze"]["EUR_USD"]["hit_rate"], 0.75)


class CalibrationTest(unittest.TestCase):
    def test_high_hit_rate_boosts_confidence(self) -> None:
        # 20 samples 100% hit
        hr = {"sig_x": {
            "EUR_USD": {"hit_rate": 1.0, "samples": 20},
            "_all_pairs": {"hit_rate": 1.0, "samples": 20},
        }}
        mult = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr)
        self.assertAlmostEqual(mult, CONFIDENCE_MAX_MULTIPLIER, places=2)

    def test_zero_hit_rate_dampens_confidence(self) -> None:
        hr = {"sig_x": {
            "EUR_USD": {"hit_rate": 0.0, "samples": 20},
        }}
        mult = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr)
        self.assertAlmostEqual(mult, CONFIDENCE_MIN_MULTIPLIER, places=2)

    def test_few_samples_returns_1_0(self) -> None:
        hr = {"sig_x": {"EUR_USD": {"hit_rate": 0.5, "samples": 3}}}
        mult = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr)
        self.assertEqual(mult, 1.0)

    def test_unknown_signal_returns_1_0(self) -> None:
        hr = {}
        mult = confidence_calibration("unknown", "EUR_USD", hit_rates=hr)
        self.assertEqual(mult, 1.0)

    def test_per_pair_takes_precedence_over_all_pairs(self) -> None:
        hr = {"sig_x": {
            "EUR_USD": {"hit_rate": 0.9, "samples": 15},
            "_all_pairs": {"hit_rate": 0.3, "samples": 100},
        }}
        # Per-pair has enough samples (≥10) → its 0.9 wins, not all-pairs 0.3
        mult = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr)
        self.assertGreater(mult, 1.0)


class SetupGradeTest(unittest.TestCase):
    def test_grade_a_strong_confluence_no_news(self) -> None:
        self.assertEqual(setup_grade(aligned_signal_count=4, has_news_block=False,
                                       confluence_score=27), "A")

    def test_grade_d_when_news_blocks(self) -> None:
        self.assertEqual(setup_grade(aligned_signal_count=10, has_news_block=True,
                                       confluence_score=50), "D")

    def test_grade_b_medium(self) -> None:
        self.assertEqual(setup_grade(aligned_signal_count=3, has_news_block=False,
                                       confluence_score=18), "B")

    def test_grade_c_minimal_confluence(self) -> None:
        self.assertEqual(setup_grade(aligned_signal_count=2, has_news_block=False,
                                       confluence_score=8), "C")


if __name__ == "__main__":
    unittest.main()
