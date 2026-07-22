from __future__ import annotations

import importlib.util
import hashlib
import gzip
import json
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.dojo_paper_contract import DojoPaperContractError
from quant_rabbit.virtual_broker import VirtualBroker


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prime_long_trend_state(bot, pair: str = "USD_JPY") -> None:
    state = bot.state[pair]
    state.closes = deque(
        [150.0 + index / 1440.0 for index in range(1440)], maxlen=1441
    )
    state.diffs_6h = deque([0.02] * 359, maxlen=360)
    state.prev_close = state.closes[-1]
    state.atr = 0.02


def _bar(epoch: int, mid: float = 151.0) -> dict:
    return {
        "epoch": epoch,
        "bid_o": mid - 0.005,
        "bid_h": mid + 0.005,
        "bid_l": mid - 0.015,
        "bid_c": mid - 0.005,
        "ask_o": mid + 0.005,
        "ask_h": mid + 0.015,
        "ask_l": mid - 0.005,
        "ask_c": mid + 0.005,
    }


def _fade_config(**overrides):
    return {
        "signal": "range_fade_limit",
        "strategy_tag": "W_FADE",
        "pairs": ["USD_JPY"],
        "tp_pips": 6,
        "sl_pips": None,
        "ceiling_min": 480,
        "max_concurrent": 3,
        "per_pos_lev": 4.3,
        "atr_floor_pips": 1.0,
        "fade_atr": 1.2,
        "eff_max": 0.2,
        **overrides,
    }


def test_lab_bot_persists_complete_indicator_context_on_every_entry_order(tmp_path):
    lab_bot = _load_module("dojo_lab_bot_context_test", ROOT / "bots" / "lab_bot.py")
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    epoch = int(datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc).timestamp())
    broker.on_quote("USD_JPY", 150.995, 151.005, "2026-07-22T00:00:00+00:00")
    bot = lab_bot.Bot(broker, _fade_config())
    _prime_long_trend_state(bot)

    bot.on_bar_closed("USD_JPY", _bar(epoch), epoch)

    assert {order.side for order in broker.orders.values()} == {"LONG", "SHORT"}
    for order in broker.orders.values():
        context = order.entry_context
        assert context is not None
        assert context["strategy_tag"] == "W_FADE"
        assert context["trend_24h"] == "LONG"
        assert context["trend_24h_change_pips"] > 0
        assert context["efficiency_6h"] <= 0.2
        assert context["atr_pips"] >= 1.0
        assert context["side"] == order.side


def test_long_trend_countertrend_short_candidate_only_blocks_short(tmp_path):
    lab_bot = _load_module("dojo_lab_bot_trend_guard_test", ROOT / "bots" / "lab_bot.py")
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    epoch = int(datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc).timestamp())
    broker.on_quote("USD_JPY", 150.995, 151.005, "2026-07-22T00:00:00+00:00")
    bot = lab_bot.Bot(broker, _fade_config(block_long_trend_short=True))
    _prime_long_trend_state(bot)

    bot.on_bar_closed("USD_JPY", _bar(epoch), epoch)

    assert [order.side for order in broker.orders.values()] == ["LONG"]


def test_daily_equity_dd_candidate_cancels_orders_and_blocks_new_entries(tmp_path):
    lab_bot = _load_module("dojo_lab_bot_dd_guard_test", ROOT / "bots" / "lab_bot.py")
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    opened = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)
    epoch = int(opened.timestamp())
    broker.on_quote("USD_JPY", 150.995, 151.005, opened.isoformat())
    bot = lab_bot.Bot(
        broker, _fade_config(daily_equity_drawdown_stop_pct=0.02)
    )
    _prime_long_trend_state(bot)
    bot.on_bar_closed("USD_JPY", _bar(epoch), epoch)
    assert len(broker.orders) == 2

    broker.market_order("USD_JPY", "LONG", 10_000)
    later = opened + timedelta(minutes=1)
    broker.on_quote("USD_JPY", 150.39, 150.40, later.isoformat())
    bot.on_bar_closed("USD_JPY", _bar(int(later.timestamp()), 150.395), int(later.timestamp()))

    assert broker.orders == {}
    records = [json.loads(line) for line in broker.ledger_path.read_text().splitlines()]
    breaker = [row for row in records if row["event"] == "ENTRY_CIRCUIT_BREAKER"]
    assert len(breaker) == 1
    assert breaker[0]["payload"]["new_entries_allowed"] is False


def test_restarted_lab_bot_keeps_original_position_ceiling(tmp_path):
    lab_bot = _load_module("dojo_lab_bot_test", ROOT / "bots" / "lab_bot.py")
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    opened = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    broker.on_quote("USD_JPY", 162.00, 162.01, opened.isoformat())
    trade_id = broker.market_order(
        "USD_JPY", "SHORT", 1_000, tp_pips=6, strategy_tag="W_FADE"
    )

    bot = lab_bot.Bot(
        broker,
        {
            "signal": "range_fade_limit",
            "strategy_tag": "W_FADE",
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


def test_restarted_combo_hand_does_not_adopt_another_strategy(tmp_path):
    lab_bot = _load_module("dojo_lab_bot_owner_test", ROOT / "bots" / "lab_bot.py")
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    opened = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    broker.on_quote("USD_JPY", 162.00, 162.01, opened.isoformat())
    foreign_trade = broker.market_order(
        "USD_JPY", "SHORT", 1_000, tp_pips=6, strategy_tag="W_SPIKE"
    )
    unknown_trade = broker.market_order("USD_JPY", "SHORT", 1_000, tp_pips=6)
    bot = lab_bot.Bot(
        broker,
        {
            "signal": "range_fade_limit",
            "strategy_tag": "W_FADE",
            "pairs": ["USD_JPY"],
            "tp_pips": 6,
            "sl_pips": None,
            "ceiling_min": 480,
            "max_concurrent": 3,
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

    assert foreign_trade in broker.positions
    assert unknown_trade in broker.positions
    assert bot.state["USD_JPY"].my_trades == {}


def test_drain_controller_only_resolves_existing_position_at_original_ceiling(
    tmp_path,
):
    runner = _load_module(
        "dojo_virtual_market_drain_test",
        ROOT / "scripts" / "run-virtual-market-session.py",
    )
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    opened = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    broker.on_quote("USD_JPY", 162.00, 162.01, opened.isoformat())
    trade_id = broker.market_order(
        "USD_JPY", "LONG", 1_000, tp_pips=200, strategy_tag="W_FADE"
    )
    controller = runner.DrainOnlyController(broker, ceiling_minutes=480)

    before = opened + timedelta(minutes=479)
    broker.on_quote("USD_JPY", 162.02, 162.03, before.isoformat())
    controller.on_quote("USD_JPY", before.isoformat())
    assert trade_id in broker.positions

    due = opened + timedelta(minutes=480)
    broker.on_quote("USD_JPY", 162.02, 162.03, due.isoformat())
    controller.on_quote("USD_JPY", due.isoformat())

    assert trade_id not in broker.positions
    events = [
        json.loads(line)["event"]
        for line in broker.ledger_path.read_text().splitlines()
    ]
    assert events[-2:] == ["DRAIN_CEILING_DUE", "CLOSE"]


def test_virtual_market_session_refuses_a_second_process_owner(tmp_path):
    runner = _load_module(
        "dojo_virtual_market_lock_test",
        ROOT / "scripts" / "run-virtual-market-session.py",
    )
    runner._acquire_runtime_lock(tmp_path)
    try:
        with pytest.raises(DojoPaperContractError, match="another virtual-market"):
            runner._acquire_runtime_lock(tmp_path)
    finally:
        runner._RUNTIME_LOCK_HANDLE.close()
        runner._RUNTIME_LOCK_HANDLE = None


def test_replay_source_manifest_binds_exact_shard_bytes(tmp_path):
    runner = _load_module(
        "dojo_virtual_market_manifest_test",
        ROOT / "scripts" / "run-virtual-market-session.py",
    )
    shard_dir = tmp_path / "run" / "USD_JPY"
    shard_dir.mkdir(parents=True)
    shard = shard_dir / "USD_JPY_M1_BA_2026.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        handle.write('{"time":"2026-01-01T00:00:00Z"}\n')
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "contract": "QR_VIRTUAL_REPLAY_SOURCE_MANIFEST_V1",
                "granularity": "M1",
                "pairs": ["USD_JPY"],
                "files": [
                    {
                        "pair": "USD_JPY",
                        "path": str(shard),
                        "sha256": digest,
                    }
                ],
            }
        )
    )

    assert runner._load_replay_source_manifest(
        manifest, pairs=["USD_JPY"], granularity="M1"
    ) == [shard.resolve()]

    with gzip.open(shard, "at", encoding="utf-8") as handle:
        handle.write('{"tampered":true}\n')
    with pytest.raises(DojoPaperContractError, match="hash mismatch"):
        runner._load_replay_source_manifest(
            manifest, pairs=["USD_JPY"], granularity="M1"
        )


def test_profitability_knowledge_separates_clean_and_carry_cohorts(tmp_path):
    knowledge = _load_module(
        "dojo_profitability_knowledge_test",
        ROOT / "scripts" / "update-dojo-profitability-knowledge.py",
    )
    session = tmp_path / "session"
    session.mkdir()
    broker = VirtualBroker(session / "ledger.jsonl", balance_jpy=200_000.0)
    opened = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)
    broker.on_quote("USD_JPY", 162.00, 162.01, opened.isoformat())
    broker.market_order(
        "USD_JPY", "LONG", 1_000, tp_pips=6, strategy_tag="W_FADE"
    )
    broker.on_quote(
        "USD_JPY", 162.07, 162.08, (opened + timedelta(minutes=5)).isoformat()
    )
    (session / "broker_snapshot.json").write_text(json.dumps(broker.snapshot()))
    (session / "state.json").write_text(
        json.dumps(
            {
                "wall_time_utc": (opened + timedelta(minutes=5)).isoformat(),
                "account": broker.account(),
                "positions": [],
            }
        )
    )

    observation = knowledge.build_observation(
        session,
        clean_after=0,
        carry_ids=set(),
        hypothesis="tagged clean evidence",
        counterevidence="one result is not sufficient",
        decision="TEST",
        next_test="collect more clean results",
    )

    assert observation["cohorts"]["all"]["settled"] == 1
    assert observation["cohorts"]["clean_postfix"]["settled"] == 1
    assert observation["cohorts"]["carry_repair"]["settled"] == 0
    assert observation["daily_2pct_benchmark"]["guarantee"] is False
