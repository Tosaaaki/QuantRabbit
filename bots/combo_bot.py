"""DOJO combo bot: several lab-bot hands sharing one broker/account.

DOJO_BOT_COMBO env var holds a JSON list of lab-bot configs; each hand
runs independently, all fills land in the same book (real portfolio
cohabitation: shared equity, shared margin, shared cage).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from quant_rabbit.virtual_broker import VirtualBroker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_bot import Bot as LabBot


class Bot:
    def __init__(self, broker: VirtualBroker):
        configs = json.loads(os.environ["DOJO_BOT_COMBO"])
        if not isinstance(configs, list) or not configs:
            raise ValueError("DOJO_BOT_COMBO must be a non-empty config list")
        normalized = []
        tags = []
        for raw in configs:
            if not isinstance(raw, dict):
                raise ValueError("each combo hand config must be an object")
            cfg = dict(raw)
            tag = str(cfg.get("strategy_tag") or cfg.get("strategy_id") or cfg["signal"])
            cfg["strategy_tag"] = tag
            normalized.append(cfg)
            tags.append(tag)
        if len(tags) != len(set(tags)):
            raise ValueError("combo strategy_tag values must be unique")
        self.hands = [LabBot(broker, cfg) for cfg in normalized]

    def seed_bar(self, pair: str, bar: dict) -> None:
        for hand in self.hands:
            if hasattr(hand, "seed_bar"):
                hand.seed_bar(pair, bar)

    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        for hand in self.hands:
            hand.on_bar_closed(pair, bar, epoch)
