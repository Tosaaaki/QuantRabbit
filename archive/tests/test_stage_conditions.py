import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.stage_rules import _stage_conditions_met


class StageConditionsGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fac_m1_base = {
            "close": 152.90,
            "ema20": 152.88,
            "ma10": 152.89,
            "ma20": 152.87,
            "adx": 18.0,
            "atr_pips": 4.0,
            "rsi": 50.0,
        }
        self.fac_h4_base = {
            "close": 152.80,
            "ma10": 152.85,
            "ma20": 152.80,
            "adx": 25.0,
        }
        self.open_info = {"avg_price": 152.90}

    def test_micro_long_blocked_by_rsi_floor(self) -> None:
        fac_m1 = dict(self.fac_m1_base)
        fac_h4 = dict(self.fac_h4_base)
        fac_m1["rsi"] = 36.0
        allowed = _stage_conditions_met(
            "micro",
            2,
            "OPEN_LONG",
            fac_m1,
            fac_h4,
            dict(self.open_info),
        )
        self.assertFalse(allowed, "RSI が高すぎるロングは拒否されるべき")

    def test_micro_long_allowed_when_thresholds_met(self) -> None:
        fac_m1 = dict(self.fac_m1_base)
        fac_h4 = dict(self.fac_h4_base)
        fac_m1["rsi"] = 33.0
        allowed = _stage_conditions_met(
            "micro",
            2,
            "OPEN_LONG",
            fac_m1,
            fac_h4,
            dict(self.open_info),
        )
        self.assertTrue(allowed, "RSI が収まっていれば通るべき")

    def test_scalp_long_blocked_by_atr_floor(self) -> None:
        fac_m1 = dict(self.fac_m1_base)
        fac_h4 = dict(self.fac_h4_base)
        fac_m1["rsi"] = 45.0
        fac_m1["atr"] = 0.015
        fac_m1["ema20"] = 152.95
        allowed = _stage_conditions_met(
            "scalp",
            1,
            "OPEN_LONG",
            fac_m1,
            fac_h4,
            dict(self.open_info),
        )
        self.assertFalse(allowed, "scalp の ATR フロア未満は拒否されるべき")


if __name__ == "__main__":
    unittest.main()
