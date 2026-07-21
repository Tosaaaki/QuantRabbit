from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.virtual_broker import VirtualBroker


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_restarted_lab_bot_keeps_original_position_ceiling(tmp_path):
    lab_bot = _load_module("dojo_lab_bot_test", ROOT / "bots" / "lab_bot.py")
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    opened = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    broker.on_quote("USD_JPY", 162.00, 162.01, opened.isoformat())
    trade_id = broker.market_order("USD_JPY", "SHORT", 1_000, tp_pips=6)

    bot = lab_bot.Bot(
        broker,
        {
            "signal": "range_fade_limit",
            "pairs": ["USD_JPY"],
            "tp_pips": 6,
            "sl_pips": None,
            "ceiling_min": 480,
            "max_concurrent": 3,
            "per_pos_lev": 4.3,
            "atr_floor_pips": 1.0,
            "fade_atr": 1.2,
            "eff_max": 0.2,
        },
    )
    current = opened + timedelta(hours=9)
    bar = {
        "epoch": int(current.timestamp()),
        "bid_o": 162.04,
        "bid_h": 162.05,
        "bid_l": 162.03,
        "bid_c": 162.04,
        "ask_o": 162.05,
        "ask_h": 162.06,
        "ask_l": 162.04,
        "ask_c": 162.05,
    }

    bot.on_bar_closed("USD_JPY", bar, bar["epoch"])

    assert trade_id not in broker.positions
    close = [
        json.loads(line)
        for line in broker.ledger_path.read_text().splitlines()
        if json.loads(line)["event"] == "CLOSE"
    ]
    assert len(close) == 1
