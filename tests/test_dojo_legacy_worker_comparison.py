from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.virtual_broker import VirtualBroker
from quant_rabbit.dojo_legacy_worker_comparison import (
    AUTHORITY,
    LegacyWorkerComparisonError,
    POLICY_CONTRACT,
    canonical_sha256,
    load_archived_candles,
    validate_policy,
)

ROOT = Path(__file__).resolve().parents[1]
BOT_SPEC = importlib.util.spec_from_file_location(
    "legacy_worker_paper_bot", ROOT / "bots/legacy_worker_paper_bot.py"
)
BOT_MODULE = importlib.util.module_from_spec(BOT_SPEC)
assert BOT_SPEC.loader is not None
BOT_SPEC.loader.exec_module(BOT_MODULE)


class LegacyWorkerComparisonTests(unittest.TestCase):
    def _policy(self) -> dict:
        return {
            "contract": POLICY_CONTRACT,
            "authority": AUTHORITY,
            "parameters": {
                "direction_lookback_bars": 15,
                "direction_block_pips": 3.0,
                "risk_fraction": 0.01,
                "high_volatility_atr_pips": 4.0,
                "high_volatility_size_multiple": 0.5,
                "inventory_same_side_cap": 1,
                "breakeven_trigger_r": 0.6,
                "partial_trigger_r": 0.8,
                "partial_fraction": 0.5,
                "trailing_trigger_r": 1.0,
                "trailing_atr_multiple": 1.0,
                "early_exit_opposition_bars": 3,
            },
        }

    def test_valid_policy_is_paper_only_and_trim_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(json.dumps(self._policy()), encoding="utf-8")
            loaded = validate_policy(path)
        self.assertEqual(loaded["authority"]["order_authority"], "NONE")
        self.assertLessEqual(
            loaded["parameters"]["high_volatility_size_multiple"], 1.0
        )

    def test_policy_rejects_live_authority(self) -> None:
        policy = self._policy()
        policy["authority"] = {**AUTHORITY, "live_permission": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaises(LegacyWorkerComparisonError):
                validate_policy(path)

    def test_candle_loader_rejects_duplicate_or_rewound_time(self) -> None:
        rows = [
            {
                "time": "2025-10-01T00:00:00Z",
                "mid": {"o": "149.0", "h": "149.1", "l": "148.9", "c": "149.0"},
            },
            {
                "time": "2025-10-01T00:01:00Z",
                "mid": {"o": "149.0", "h": "149.2", "l": "148.9", "c": "149.1"},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candles.json"
            path.write_text(json.dumps({"candles": rows}), encoding="utf-8")
            loaded = load_archived_candles([path])
        self.assertEqual(len(loaded), 2)
        self.assertEqual(
            canonical_sha256({"authority": AUTHORITY}),
            canonical_sha256({"authority": dict(AUTHORITY)}),
        )

    def test_paper_bot_rejects_live_authority_before_registering_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broker = VirtualBroker(
                ledger_path=Path(tmp) / "ledger.jsonl",
                balance_jpy=200_000.0,
            )
            config = {
                "authority": {**AUTHORITY, "live_permission": True},
                "family": "PulseBreak",
                "management_arm": "BOT_ONLY",
                "strategy_owner_id": "test-owner",
                "pairs": ["USD_JPY"],
                "risk_fraction": 0.01,
                "tp_pips": 6.4,
                "sl_pips": 4.48,
                "ceiling_bars": 7,
            }
            with self.assertRaises(ValueError):
                BOT_MODULE.Bot(broker, config)


if __name__ == "__main__":
    unittest.main()
