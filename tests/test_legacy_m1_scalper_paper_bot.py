from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.dojo_legacy_worker_comparison import AUTHORITY
from quant_rabbit.virtual_broker import VirtualBroker


ROOT = Path(__file__).resolve().parents[1]
BOT_SPEC = importlib.util.spec_from_file_location(
    "legacy_m1_scalper_paper_bot",
    ROOT / "bots/legacy_m1_scalper_paper_bot.py",
)
BOT_MODULE = importlib.util.module_from_spec(BOT_SPEC)
assert BOT_SPEC.loader is not None
BOT_SPEC.loader.exec_module(BOT_MODULE)


class LegacyM1PaperBotTests(unittest.TestCase):
    def _config(self, arm: str) -> dict:
        return {
            "authority": AUTHORITY,
            "management_arm": arm,
            "strategy_owner_id": f"test-m1-{arm.lower()}",
            "operation_id": f"dojo-m1scalper-paper:test-{arm.lower()}:v1",
            "pairs": ["USD_JPY"],
            "fixed_units": 1000,
            "ceiling_bars": 10,
            "ai_policy_path": str(
                ROOT / "config/dojo_legacy_m1_ai_inventory_policy_v1.json"
            ),
        }

    def _bot(self, root: Path, arm: str):
        broker = VirtualBroker(
            ledger_path=root / f"{arm}.broker.jsonl",
            balance_jpy=200_000.0,
            slippage_pips=0.2,
        )
        decision = root / f"{arm}.decisions.jsonl"
        with patch.dict(
            os.environ, {"DOJO_M1_DECISION_LEDGER": str(decision)}, clear=False
        ):
            bot = BOT_MODULE.Bot(broker, self._config(arm))
        return broker, bot, decision

    def test_live_authority_is_rejected_before_owner_registration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broker = VirtualBroker(
                ledger_path=root / "broker.jsonl", balance_jpy=200_000.0
            )
            config = self._config("BOT_ONLY")
            config["authority"] = {**AUTHORITY, "live_permission": True}
            with patch.dict(
                os.environ,
                {"DOJO_M1_DECISION_LEDGER": str(root / "decision.jsonl")},
                clear=False,
            ):
                with self.assertRaises(ValueError):
                    BOT_MODULE.Bot(broker, config)

    def test_fixed_units_above_legacy_cap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broker = VirtualBroker(
                ledger_path=root / "broker.jsonl", balance_jpy=200_000.0
            )
            config = self._config("BOT_ONLY")
            config["fixed_units"] = 1001
            with patch.dict(
                os.environ,
                {"DOJO_M1_DECISION_LEDGER": str(root / "decision.jsonl")},
                clear=False,
            ):
                with self.assertRaises(ValueError):
                    BOT_MODULE.Bot(broker, config)

    def test_bot_and_ai_ledgers_have_distinct_operation_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, bot_ledger = self._bot(root, "BOT_ONLY")
            _, _, ai_ledger = self._bot(root, "AI_INVENTORY")
            bot_row = json.loads(bot_ledger.read_text(encoding="utf-8").splitlines()[0])
            ai_row = json.loads(ai_ledger.read_text(encoding="utf-8").splitlines()[0])
            self.assertNotEqual(bot_row["operation_id"], ai_row["operation_id"])
            self.assertNotEqual(
                bot_row["room_operation_id"], ai_row["room_operation_id"]
            )
            self.assertEqual(bot_row["authority"]["order_authority"], "NONE")
            self.assertFalse(bot_row["detail"]["lot_increase_allowed"])

    def test_ai_blocks_disallowed_session_without_paper_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broker, bot, decision = self._bot(Path(tmp), "AI_INVENTORY")
            broker.on_quote_batch(
                [("USD_JPY", 150.0, 150.01, "2026-07-29T12:00:00+00:00")]
            )
            bot._submit(
                {
                    "action": "OPEN_LONG",
                    "tp_pips": 5.0,
                    "sl_pips": 9.0,
                    "confidence": 70,
                    "tag": "test",
                },
                1_785_326_400,
            )
            self.assertEqual(broker.positions, {})
            self.assertEqual(broker.orders, {})
            rows = [
                json.loads(line)
                for line in decision.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[-1]["action"], "AI_ENTRY_DECISION")
            self.assertFalse(rows[-1]["detail"]["session_direction_allowed"])

    def test_bot_submission_uses_exact_fixed_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broker, bot, _ = self._bot(Path(tmp), "BOT_ONLY")
            broker.on_quote_batch(
                [("USD_JPY", 150.0, 150.01, "2026-07-29T23:00:00+00:00")]
            )
            bot._submit(
                {
                    "action": "OPEN_LONG",
                    "tp_pips": 5.0,
                    "sl_pips": 9.0,
                    "confidence": 70,
                    "tag": "test",
                },
                1_785_366_000,
            )
            position = next(iter(broker.positions.values()))
            self.assertEqual(position.units, 1000)


if __name__ == "__main__":
    unittest.main()
