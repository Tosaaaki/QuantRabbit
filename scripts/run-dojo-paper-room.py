#!/usr/bin/env python3
"""Launch one pre-registered, isolated DOJO forward-paper room."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_CONTRACT = "QR_DOJO_PAPER_ROOM_REGISTRY_V1"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._@-]{0,95}$")


class RoomRegistryError(ValueError):
    pass


def load_room(registry_path: Path, room_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoomRegistryError(f"invalid room registry: {registry_path}") from exc
    if registry.get("contract") != REGISTRY_CONTRACT:
        raise RoomRegistryError("unsupported room registry contract")
    if not SAFE_ID.fullmatch(room_id):
        raise RoomRegistryError(f"unsafe room id: {room_id}")
    rooms = registry.get("rooms")
    if not isinstance(rooms, list) or not rooms:
        raise RoomRegistryError("registry rooms must be a non-empty list")
    ids = [row.get("room_id") for row in rooms if isinstance(row, dict)]
    if len(ids) != len(rooms) or len(ids) != len(set(ids)):
        raise RoomRegistryError("registry room ids must be present and unique")
    try:
        room = next(row for row in rooms if row["room_id"] == room_id)
    except StopIteration as exc:
        raise RoomRegistryError(f"room is not registered: {room_id}") from exc
    return registry, room


def build_launch(
    *, registry_path: Path, room_id: str, python_executable: str
) -> tuple[list[str], dict[str, str], Path]:
    registry, room = load_room(registry_path, room_id)
    defaults = registry.get("defaults") or {}
    window = registry.get("window") or {}
    costs = room.get("costs") or {}
    config = room.get("bot_config")
    if not isinstance(config, dict):
        raise RoomRegistryError("room bot_config must be an object")
    strategy_tag = config.get("strategy_tag")
    if not isinstance(strategy_tag, str) or not strategy_tag:
        raise RoomRegistryError("single-strategy room requires a strategy_tag")
    if room.get("arm") not in {"BASE", "STRESS"}:
        raise RoomRegistryError("room arm must be BASE or STRESS")
    required = {
        "experiment_id": registry.get("experiment_id"),
        "candidate_id": room.get("candidate_id"),
        "window_start": window.get("start_utc"),
        "window_end": window.get("end_utc"),
        "bot_module": defaults.get("bot_module"),
        "bot_config_env": defaults.get("bot_config_env"),
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        raise RoomRegistryError("missing registry fields: " + ",".join(missing))
    pairs = defaults.get("pairs")
    if not isinstance(pairs, list) or not pairs or len(pairs) != len(set(pairs)):
        raise RoomRegistryError("default pairs must be non-empty and unique")
    if sorted(config.get("pairs") or []) != sorted(pairs):
        raise RoomRegistryError("room bot pairs must exactly match feed pairs")

    experiment_id = str(required["experiment_id"])
    if not SAFE_ID.fullmatch(experiment_id):
        raise RoomRegistryError(f"unsafe experiment id: {experiment_id}")
    session_dir = (
        REPO_ROOT
        / "research/data/dojo_paper_rooms_v1"
        / experiment_id
        / room_id
    )
    seed_root = REPO_ROOT / str(defaults.get("seed_m1_root"))
    bot_module = REPO_ROOT / str(required["bot_module"]).partition(":")[0]
    bot_class = str(required["bot_module"]).partition(":")[2] or "Bot"
    bot_spec = f"{bot_module}:{bot_class}"
    command = [
        python_executable,
        str(REPO_ROOT / "scripts/run-virtual-market-session.py"),
        "--feed",
        "live",
        "--session-dir",
        str(session_dir),
        "--pairs",
        ",".join(pairs),
        "--balance",
        str(float(defaults["balance_jpy"])),
        "--window-start-utc",
        str(required["window_start"]),
        "--window-end-utc",
        str(required["window_end"]),
        "--bot-module",
        bot_spec,
        "--bot-config-env",
        str(required["bot_config_env"]),
        "--seed-m1-root",
        str(seed_root),
        "--seed-hours",
        str(float(defaults["seed_hours"])),
        "--slippage-pips",
        str(float(costs["slippage_pips_per_fill"])),
        "--financing-pips-day",
        str(float(costs["financing_pips_per_day"])),
        "--leverage",
        str(float(defaults["leverage"])),
        "--paper-proof-mode",
        "formal",
        "--room-kind",
        "single_strategy",
        "--experiment-id",
        experiment_id,
        "--room-id",
        room_id,
        "--candidate-id",
        str(required["candidate_id"]),
        "--runtime-dependency",
        str(registry_path.resolve()),
        "--runtime-dependency",
        str(Path(__file__).resolve()),
    ]
    env = dict(os.environ)
    env[str(required["bot_config_env"])] = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.setdefault(
        "QR_OANDA_ENV_FILE", "/Users/tossaki/App/QuantRabbit-live/.env.local"
    )
    return command, env, session_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "config/dojo_paper_rooms_v1.json",
    )
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command, env, session_dir = build_launch(
        registry_path=args.registry.resolve(),
        room_id=args.room_id,
        python_executable=sys.executable,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "room_id": args.room_id,
                    "session_dir": str(session_dir),
                    "bot_config": json.loads(env["DOJO_BOT_CONFIG"]),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    session_dir.mkdir(parents=True, exist_ok=True)
    os.execvpe(command[0], command, env)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
