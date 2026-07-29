from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

from quant_rabbit.dojo_legacy_m1_signal import (
    CausalM1Signal,
    LegacyM1SignalError,
)
from quant_rabbit.legacy_m1_frozen import (
    SOURCE_INDICATOR_SHA256,
    SOURCE_STRATEGY_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "src/quant_rabbit/legacy_m1_frozen"


def _bar(index: int) -> dict:
    base = 150.0 + 0.0008 * index + 0.025 * math.sin(index / 3)
    close = base + 0.006 * math.sin(index)
    return {
        "epoch": 1_700_000_000 + index * 60,
        "bid_o": base - 0.005,
        "ask_o": base + 0.005,
        "bid_h": base + 0.025,
        "ask_h": base + 0.035,
        "bid_l": base - 0.025,
        "ask_l": base - 0.015,
        "bid_c": close,
        "ask_c": close + 0.01,
    }


class LegacyM1SignalTests(unittest.TestCase):
    def test_frozen_sources_match_commit_except_two_declared_repairs(self) -> None:
        indicator = (FROZEN / "calc_core_d8f751afc.py").read_bytes()
        self.assertEqual(hashlib.sha256(indicator).hexdigest(), SOURCE_INDICATOR_SHA256)

        port = (FROZEN / "m1_scalper_d8f751afc.py").read_text(encoding="utf-8")
        port = port.replace(
            "_CONFIG_PATH = (\n"
            "    Path(__file__).resolve().parents[3]\n"
            '    / "config"\n'
            '    / "dojo_legacy_m1_scalper_d8f751afc.json"\n'
            ")\n",
            '_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scalp_active_params.json"\n',
        )
        port = port.replace(
            "    # Causal repair for replay/Paper: the legacy implementation consulted the\n"
            "    # process wall clock.  The adapter injects the completed bar's UTC hour,\n"
            "    # which is identical in forward operation and prevents replay lookahead.\n"
            '    hour = int(fac.get("_legacy_utc_hour", time.gmtime().tm_hour))\n',
            "    hour = time.gmtime().tm_hour\n",
        )
        self.assertEqual(
            hashlib.sha256(port.encode("utf-8")).hexdigest(),
            SOURCE_STRATEGY_SHA256,
        )

    def test_golden_signal_matches_frozen_reference_behavior(self) -> None:
        signal = CausalM1Signal()
        observed = None
        for index in range(133):
            observed = signal.add_completed_bar(_bar(index), emit_signal=True)
        self.assertEqual(
            observed,
            {
                "action": "OPEN_SHORT",
                "sl_pips": 8.55,
                "tp_pips": 8.74,
                "confidence": 73,
                "fast_cut_pips": 4.25,
                "fast_cut_time_sec": 50,
                "fast_cut_hard_mult": 1.6,
                "tag": "M1Scalper-sell-rally",
                "notes": {"tech_mult": 0.971},
                "exit_tags": ["kill", "fast_cut"],
                "kill_switch": True,
            },
        )

    def test_future_bar_changes_cannot_change_an_emitted_decision(self) -> None:
        left = CausalM1Signal()
        right = CausalM1Signal()
        left_decision = right_decision = None
        for index in range(133):
            left_decision = left.add_completed_bar(_bar(index), emit_signal=True)
            right_decision = right.add_completed_bar(_bar(index), emit_signal=True)
        future = _bar(133)
        future.update(
            {
                "bid_h": 999.0,
                "ask_h": 999.01,
                "bid_l": 1.0,
                "ask_l": 1.01,
            }
        )
        right.add_completed_bar(future, emit_signal=True)
        self.assertEqual(left_decision, right_decision)

    def test_duplicate_or_rewound_completed_bar_is_rejected(self) -> None:
        signal = CausalM1Signal()
        signal.seed_bar(_bar(0))
        with self.assertRaises(LegacyM1SignalError):
            signal.seed_bar(_bar(0))

    def test_config_is_the_exact_m1_section_used_by_source_commit(self) -> None:
        payload = json.loads(
            (ROOT / "config/dojo_legacy_m1_scalper_d8f751afc.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["M1Scalper"]["fallback"]["atr_floor"], 1.6)
        self.assertEqual(payload["M1Scalper"]["nwave"]["hard_sl_floor"], 7.2)


if __name__ == "__main__":
    unittest.main()
