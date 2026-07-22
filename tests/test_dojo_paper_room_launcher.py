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
