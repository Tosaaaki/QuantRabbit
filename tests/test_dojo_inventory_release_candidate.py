from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pytest

from bots.inventory_release_candidate import Bot
from quant_rabbit.virtual_broker import VirtualBroker


def _bar(epoch: int, close: float) -> dict:
    return {
        "epoch": epoch,
        "bid_o": close - 0.004,
        "bid_h": close + 0.006,
        "bid_l": close - 0.006,
        "bid_c": close - 0.004,
        "ask_o": close + 0.004,
        "ask_h": close + 0.006,
        "ask_l": close - 0.006,
        "ask_c": close + 0.004,
    }


def _load_replay_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run-dojo-inventory-release-replay.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dojo_inventory_release_replay", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_metrics_use_market_quote_time_not_append_wall_clock(tmp_path):
    replay = _load_replay_module()
    rows = [
        {
            "event": "FILL_LIMIT",
            "ts_utc": "2026-07-23T11:51:04.000001+00:00",
            "payload": {
                "trade_id": "T1",
                "quote": {
                    "ts": "2026-01-05T00:01:00+00:00#H",
                },
            },
        },
        {
            "event": "CLOSE",
            "ts_utc": "2026-07-23T11:51:04.000002+00:00",
            "payload": {
                "trade_id": "T1",
                "pl_jpy": -100.0,
                "quote": {
                    "ts": "2026-01-05T01:01:00+00:00#C",
                },
            },
        },
    ]
    (tmp_path / "ledger.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "broker_snapshot.json").write_text(
        json.dumps({"positions": [], "orders": [], "ledger_sha": "a" * 64}),
        encoding="utf-8",
    )

    metrics = replay.ledger_metrics(tmp_path)

    assert metrics["average_hold_minutes"] == 60.0
    assert metrics["active_days"] == 1
    assert metrics["measurement_clock"] == "payload.quote.ts"


def test_replay_market_time_fails_closed_without_quote_timestamp():
    replay = _load_replay_module()
    with pytest.raises(ValueError, match="payload.quote.ts"):
        replay.market_event_time(
            {
                "event": "CLOSE",
                "ts_utc": "2026-07-23T11:51:04+00:00",
                "payload": {"trade_id": "T1", "pl_jpy": -1.0},
            }
        )


def _config() -> dict:
    return {
        "signal": "prev_day_extreme_fade",
        "pairs": ["USD_JPY"],
        "tp_atr": 3.0,
        "sl_pips": 25.0,
        "ceiling_min": 60,
        "max_concurrent": 1,
        "global_max_concurrent": 1,
        "per_pos_lev": 2.0,
        "atr_floor_pips": 0.5,
        "strategy_tag": "TEST_PREV_DAY",
        "inventory_release_min_age_min": 30,
        "inventory_release_efficiency_min": 0.25,
    }


def _broker(tmp_path) -> VirtualBroker:
    broker = VirtualBroker(
        ledger_path=tmp_path / "ledger.jsonl",
        balance_jpy=200_000,
        leverage=25,
        slippage_pips=0,
        financing_pips_per_day=0,
        fast_ledger=True,
    )
    broker.on_quote(
        "USD_JPY", 163.00, 163.01, "2026-01-02T00:00:00+00:00"
    )
    return broker


def test_candidate_releases_only_mature_opposed_efficient_inventory(tmp_path):
    broker = _broker(tmp_path)
    bot = Bot(broker, _config())
    start = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    for index in range(1441):
        bot.seed_bar("USD_JPY", _bar(start + index * 60, 160 + index * 0.002))
    trade_id = broker.market_order(
        "USD_JPY",
        "SHORT",
        1_000,
        strategy_tag="TEST_PREV_DAY",
        entry_context={
            "contract": "QR_DOJO_ENTRY_CONTEXT_V1",
            "strategy_tag": "TEST_PREV_DAY",
            "signal": "prev_day_extreme_fade",
            "pair": "USD_JPY",
            "decision_bar_epoch": start + 1440 * 60,
            "decision_bar_ts_utc": "2026-01-02T00:00:00+00:00",
            "trend_24h": "LONG",
            "trend_24h_change_pips": 288.0,
            "change_6h_pips": 72.0,
            "efficiency_6h": 1.0,
            "atr_pips": 1.0,
            "side": "SHORT",
        },
    )
    opened = start + 1440 * 60
    bot.state["USD_JPY"].my_trades[trade_id] = opened
    bot._owner[trade_id] = "USD_JPY"
    broker.on_quote(
        "USD_JPY",
        163.10,
        163.11,
        datetime.fromtimestamp(opened + 29 * 60, timezone.utc).isoformat(),
    )
    bot.on_bar_closed(
        "USD_JPY", _bar(opened + 29 * 60, 163.10), opened + 29 * 60
    )
    assert trade_id in broker.positions
    broker.on_quote(
        "USD_JPY",
        163.12,
        163.13,
        datetime.fromtimestamp(opened + 30 * 60, timezone.utc).isoformat(),
    )
    bot.on_bar_closed(
        "USD_JPY", _bar(opened + 30 * 60, 163.12), opened + 30 * 60
    )
    assert trade_id not in broker.positions
    events = [
        __import__("json").loads(line)["event"]
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]
    assert "INVENTORY_RELEASE_DECISION" in events
    assert "CLOSE" in events


def test_candidate_keeps_direction_aligned_inventory(tmp_path):
    broker = _broker(tmp_path)
    bot = Bot(broker, _config())
    start = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    for index in range(1441):
        bot.seed_bar("USD_JPY", _bar(start + index * 60, 160 + index * 0.002))
    trade_id = broker.market_order(
        "USD_JPY",
        "LONG",
        1_000,
        strategy_tag="TEST_PREV_DAY",
        entry_context={
            "contract": "QR_DOJO_ENTRY_CONTEXT_V1",
            "strategy_tag": "TEST_PREV_DAY",
            "signal": "prev_day_extreme_fade",
            "pair": "USD_JPY",
            "decision_bar_epoch": start + 1440 * 60,
            "decision_bar_ts_utc": "2026-01-02T00:00:00+00:00",
            "trend_24h": "LONG",
            "trend_24h_change_pips": 288.0,
            "change_6h_pips": 72.0,
            "efficiency_6h": 1.0,
            "atr_pips": 1.0,
            "side": "LONG",
        },
    )
    opened = start + 1440 * 60
    bot.state["USD_JPY"].my_trades[trade_id] = opened
    bot._owner[trade_id] = "USD_JPY"
    broker.on_quote(
        "USD_JPY",
        163.12,
        163.13,
        datetime.fromtimestamp(opened + 30 * 60, timezone.utc).isoformat(),
    )
    bot.on_bar_closed(
        "USD_JPY", _bar(opened + 30 * 60, 163.12), opened + 30 * 60
    )
    assert trade_id in broker.positions
