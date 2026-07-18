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

from quant_rabbit.dojo_lab_provenance import canonical_strategy_owner_id
from quant_rabbit.virtual_broker import VirtualBroker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_bot import Bot as LabBot


class Bot:
    def __init__(self, broker: VirtualBroker):
        configs = json.loads(os.environ["DOJO_BOT_COMBO"])
        owned_configs = []
        for index, raw_cfg in enumerate(configs):
            cfg = dict(raw_cfg)
            cfg.setdefault(
                "strategy_owner_id",
                canonical_strategy_owner_id(cfg, namespace="combo", ordinal=index),
            )
            owned_configs.append(cfg)
        owner_ids = [str(cfg["strategy_owner_id"]) for cfg in owned_configs]
        if len(set(owner_ids)) != len(owner_ids):
            raise ValueError("DOJO combo strategy_owner_id values must be unique")
        self.hands = [LabBot(broker, cfg) for cfg in owned_configs]

    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        for hand in self.hands:
            hand.on_bar_closed(pair, bar, epoch)
