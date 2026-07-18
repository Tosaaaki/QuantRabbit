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
        self.hands = [LabBot(broker, cfg) for cfg in configs]

    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        for hand in self.hands:
            hand.on_bar_closed(pair, bar, epoch)
