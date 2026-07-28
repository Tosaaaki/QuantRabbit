from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.virtual_broker import VirtualBroker


ROOT = Path(__file__).resolve().parents[1]
BOT_SPEC = importlib.util.spec_from_file_location(
    "dojo_paper_direction_bot", ROOT / "bots/lab_bot.py"
)
BOT_MODULE = importlib.util.module_from_spec(BOT_SPEC)
assert BOT_SPEC.loader is not None
BOT_SPEC.loader.exec_module(BOT_MODULE)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "dojo_paper_room_runner", ROOT / "scripts/run-dojo-paper-room.py"
)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(RUNNER)
SESSION_SPEC = importlib.util.spec_from_file_location(
    "dojo_paper_virtual_session", ROOT / "scripts/run-virtual-market-session.py"
)
SESSION = importlib.util.module_from_spec(SESSION_SPEC)
assert SESSION_SPEC.loader is not None
SESSION_SPEC.loader.exec_module(SESSION)


def _config(policy: str, owner: str) -> dict:
    return {
        "strategy_owner_id": owner,
        "signal": "range_fade_limit",
        "pairs": ["USD_JPY"],
        "tp_pips": 6.0,
        "sl_pips": None,
        "ceiling_min": 480,
        "max_concurrent": 3,
        "max_concurrent_per_pair": 3,
        "global_max_concurrent": 3,
        "per_pos_lev": 4.3,
        "atr_floor_pips": 1.0,
        "fade_atr": 1.2,
        "eff_max": 0.2,
        "entry_direction_policy": policy,
        "external_broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
    }


def _bar(epoch: int, ordinal: int) -> dict:
    drift = ordinal * 0.00001
    oscillation = 0.03 if ordinal % 2 else -0.03
    close = 160.0 + drift + oscillation
    return {
        "epoch": epoch,
        "bid_o": close - 0.01,
        "bid_h": close + 0.02,
        "bid_l": close - 0.02,
        "bid_c": close,
        "ask_o": close,
        "ask_h": close + 0.03,
        "ask_l": close - 0.01,
        "ask_c": close + 0.01,
    }


def _sides_after_one_decision(tmp_path: Path, policy: str) -> set[str]:
    broker = VirtualBroker(
        ledger_path=tmp_path / f"{policy}.jsonl",
        balance_jpy=200_000.0,
    )
    bot = BOT_MODULE.Bot(broker, _config(policy, policy.lower()))
    start = 1_785_200_000
    for ordinal in range(1500):
        bot.seed_bar("USD_JPY", _bar(start + ordinal * 60, ordinal))
    decision = _bar(start + 1500 * 60, 1500)
    broker.on_quote(
        "USD_JPY",
        decision["bid_c"],
        decision["ask_c"],
        datetime.fromtimestamp(decision["epoch"], timezone.utc).isoformat(),
    )
    bot.on_bar_closed("USD_JPY", decision, decision["epoch"])
    return {order.side for order in broker.orders.values()}


def test_direction_gate_removes_only_completed_bar_countertrend_side(
    tmp_path: Path,
) -> None:
    assert _sides_after_one_decision(tmp_path, "BOTH_SIDES") == {"LONG", "SHORT"}
    assert _sides_after_one_decision(
        tmp_path, "FOLLOW_24H_TREND"
    ) == {"LONG"}


def test_direction_gate_rejects_unreviewed_family(tmp_path: Path) -> None:
    broker = VirtualBroker(ledger_path=tmp_path / "bad.jsonl", balance_jpy=200_000.0)
    config = _config("FOLLOW_24H_TREND", "bad-family")
    config["signal"] = "spike_fade"
    config.pop("fade_atr")
    config.pop("eff_max")
    config["tp_atr"] = 3.0
    config.pop("tp_pips")
    with pytest.raises(ValueError, match="only for range_fade_limit"):
        BOT_MODULE.Bot(broker, config)


def test_registry_builds_four_authority_none_live_read_only_rooms() -> None:
    registry_path = ROOT / "config/dojo_paper_direction_pair_20260728_v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert len(registry["rooms"]) == 4
    for room in registry["rooms"]:
        command, env, session_dir = RUNNER.build_launch(
            registry_path=registry_path,
            room_id=room["room_id"],
            python_executable="/usr/bin/python3",
            now_utc=datetime(2026, 7, 28, 7, 0, tzinfo=timezone.utc),
        )
        assert command[command.index("--feed") + 1] == "live"
        assert command[command.index("--seed-oanda-m1-count") + 1] == "1500"
        assert session_dir.parts[-2:] == (
            registry["experiment_id"],
            room["room_id"],
        )
        config = json.loads(env["DOJO_BOT_CONFIG"])
        assert config["external_broker_mutation_allowed"] is False
        assert config["live_permission"] is False
        assert config["order_authority"] == "NONE"


def test_read_only_seed_uses_only_complete_past_candles() -> None:
    now = datetime(2026, 7, 28, 7, 0, tzinfo=timezone.utc)
    candles = []
    for ordinal in range(1501):
        stamp = now - timedelta(minutes=1501 - ordinal)
        price = 150.0 + ordinal * 0.00001
        candles.append(
            {
                "complete": True,
                "time": stamp.isoformat().replace("+00:00", "Z"),
                "bid": {
                    "o": str(price),
                    "h": str(price + 0.01),
                    "l": str(price - 0.01),
                    "c": str(price),
                },
                "ask": {
                    "o": str(price + 0.01),
                    "h": str(price + 0.02),
                    "l": str(price),
                    "c": str(price + 0.01),
                },
            }
        )
    candles.append(
        {
            "complete": False,
            "time": now.isoformat().replace("+00:00", "Z"),
        }
    )

    class ReadOnlyClient:
        def get_json(self, path: str, params: dict) -> dict:
            assert path == "/v3/instruments/USD_JPY/candles"
            assert params == {"price": "BA", "granularity": "M1", "count": "1502"}
            return {"candles": candles}

    class SeedOnlyBot:
        def __init__(self) -> None:
            self.bars = []

        def seed_bar(self, pair: str, bar: dict) -> None:
            assert pair == "USD_JPY"
            self.bars.append(bar)

    bot = SeedOnlyBot()
    receipt = SESSION._seed_bot_from_read_only_oanda(
        ReadOnlyClient(),
        bot,
        ["USD_JPY"],
        count=1502,
        now_utc=now,
    )
    assert len(bot.bars) == 1501
    assert all(bar["epoch"] < int(now.timestamp()) for bar in bot.bars)
    assert receipt["source"] == "OANDA_INSTRUMENT_CANDLES_GET_ONLY"
    assert receipt["broker_mutation_allowed"] is False
    assert receipt["live_permission"] is False
    assert receipt["order_authority"] == "NONE"
