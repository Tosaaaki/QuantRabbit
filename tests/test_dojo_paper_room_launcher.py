from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_launcher():
    path = ROOT / "scripts/run-dojo-paper-room.py"
    spec = importlib.util.spec_from_file_location("dojo_paper_room_launcher_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_wave():
    path = ROOT / "scripts/run-dojo-paper-wave.py"
    spec = importlib.util.spec_from_file_location("dojo_paper_wave_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_drain_launcher():
    path = ROOT / "scripts/run-dojo-paper-room-drain.py"
    spec = importlib.util.spec_from_file_location("dojo_paper_room_drain_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registry_builds_four_isolated_formal_rooms_with_explicit_cost_arms():
    launcher = _load_launcher()
    registry_path = ROOT / "config/dojo_paper_rooms_v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    observed = {}

    for room in registry["rooms"]:
        command, env, session_dir = launcher.build_launch(
            registry_path=registry_path,
            room_id=room["room_id"],
            python_executable="/fixed/python3",
        )
        assert "--paper-proof-mode" in command
        assert command[command.index("--paper-proof-mode") + 1] == "formal"
        assert command[command.index("--room-kind") + 1] == "single_strategy"
        assert command[command.index("--window-start-utc") + 1] == registry[
            "window"
        ]["start_utc"]
        assert command[command.index("--window-end-utc") + 1] == registry[
            "window"
        ]["end_utc"]
        assert str(registry_path.resolve()) in command
        assert room["room_id"] in str(session_dir)
        config = json.loads(env["DOJO_BOT_CONFIG"])
        assert config["strategy_tag"] in {"W_FADE", "W_SPIKE"}
        observed[(config["strategy_tag"], room["arm"])] = (
            float(command[command.index("--slippage-pips") + 1]),
            float(command[command.index("--financing-pips-day") + 1]),
        )

    assert observed == {
        ("W_FADE", "BASE"): (0.0, 0.0),
        ("W_FADE", "STRESS"): (0.3, 0.8),
        ("W_SPIKE", "BASE"): (0.0, 0.0),
        ("W_SPIKE", "STRESS"): (0.3, 0.8),
    }


def test_registry_refuses_duplicate_room_ids(tmp_path):
    launcher = _load_launcher()
    source = json.loads(
        (ROOT / "config/dojo_paper_rooms_v1.json").read_text(encoding="utf-8")
    )
    source["rooms"][1]["room_id"] = source["rooms"][0]["room_id"]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(launcher.RoomRegistryError, match="unique"):
        launcher.load_room(path, source["rooms"][0]["room_id"])


def test_smoke_registry_can_be_explicitly_diagnostic(tmp_path):
    launcher = _load_launcher()
    source = json.loads(
        (ROOT / "config/dojo_paper_rooms_v1.json").read_text(encoding="utf-8")
    )
    source["proof_mode"] = "diagnostic"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    command, _, _ = launcher.build_launch(
        registry_path=path,
        room_id=source["rooms"][0]["room_id"],
        python_executable="/fixed/python3",
    )

    assert command[command.index("--paper-proof-mode") + 1] == "diagnostic"


def test_non_jpy_room_can_add_conversion_feed_without_bot_trading_it(tmp_path):
    launcher = _load_launcher()
    source = json.loads(
        (ROOT / "config/dojo_paper_rooms_v1.json").read_text(encoding="utf-8")
    )
    source["defaults"]["pairs"] = ["EUR_USD"]
    source["defaults"]["feed_pairs"] = ["EUR_USD", "USD_JPY"]
    for room in source["rooms"]:
        room["bot_config"]["pairs"] = ["EUR_USD"]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    command, env, _ = launcher.build_launch(
        registry_path=path,
        room_id=source["rooms"][0]["room_id"],
        python_executable="/fixed/python3",
    )

    assert command[command.index("--pairs") + 1] == "EUR_USD,USD_JPY"
    assert json.loads(env["DOJO_BOT_CONFIG"])["pairs"] == ["EUR_USD"]


def test_registry_refuses_conversion_feed_that_omits_bot_pair(tmp_path):
    launcher = _load_launcher()
    source = json.loads(
        (ROOT / "config/dojo_paper_rooms_v1.json").read_text(encoding="utf-8")
    )
    source["defaults"]["feed_pairs"] = ["EUR_USD"]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(launcher.RoomRegistryError, match="present in feed_pairs"):
        launcher.build_launch(
            registry_path=path,
            room_id=source["rooms"][0]["room_id"],
            python_executable="/fixed/python3",
        )


def test_non_jpy_room_drain_keeps_conversion_feed(tmp_path):
    drain = _load_drain_launcher()
    source = json.loads(
        (ROOT / "config/dojo_paper_rooms_v1.json").read_text(encoding="utf-8")
    )
    source["defaults"]["pairs"] = ["EUR_USD"]
    source["defaults"]["feed_pairs"] = ["EUR_USD", "USD_JPY"]
    for room in source["rooms"]:
        room["bot_config"]["pairs"] = ["EUR_USD"]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    command, _, _ = drain.build_drain_launch(
        registry_path=path,
        room_id=source["rooms"][0]["room_id"],
        python_executable="/fixed/python3",
    )

    assert command[command.index("--pairs") + 1] == "EUR_USD,USD_JPY"


def test_wave_uses_one_detached_owner_name_and_session_per_room():
    wave = _load_wave()
    registry_path = ROOT / "config/dojo_paper_rooms_v1.json"
    rows = wave.build_wave(registry_path, "/fixed/python3")

    assert len(rows) == 4
    assert len({row["screen_name"] for row in rows}) == 4
    assert len({str(row["session_dir"]) for row in rows}) == 4
    for row in rows:
        assert row["screen_name"] == f"qr-dojo-{row['room_id']}"
        assert row["command"][0] == "/fixed/python3"
        assert row["command"][-2:] == ["--room-id", row["room_id"]]


def test_wave_screen_command_uses_macos_compatible_detached_owner():
    wave = _load_wave()
    row = {
        "screen_name": "qr-dojo-room-01",
        "command": ["/fixed/python3", "/fixed/runner.py"],
    }

    command = wave.build_screen_command(row)

    assert command == [
        "screen",
        "-dmS",
        "qr-dojo-room-01",
        "/fixed/python3",
        "/fixed/runner.py",
    ]
    assert "-Logfile" not in command


@pytest.mark.parametrize(
    ("room_id", "slippage", "financing"),
    [
        ("room-01-w-fade-base", "0.0", "0.0"),
        ("room-02-w-fade-stress", "0.3", "0.8"),
        ("room-03-w-spike-base", "0.0", "0.0"),
        ("room-04-w-spike-stress", "0.3", "0.8"),
    ],
)
def test_room_drain_inherits_costs_and_has_no_entry_worker(
    room_id, slippage, financing
):
    drain = _load_drain_launcher()
    command, _, _ = drain.build_drain_launch(
        registry_path=ROOT / "config/dojo_paper_rooms_v1.json",
        room_id=room_id,
        python_executable="/fixed/python3",
    )

    assert "--drain-only" in command
    assert "--bot" not in command
    assert "--bot-module" not in command
    assert "--allow-legacy-untagged" not in command
    assert command[command.index("--slippage-pips") + 1] == slippage
    assert command[command.index("--financing-pips-day") + 1] == financing
    assert command[command.index("--drain-ceiling-min") + 1] == "480"
