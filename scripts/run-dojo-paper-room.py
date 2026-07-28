#!/usr/bin/env python3
"""Launch one preregistered prospective DOJO Paper room."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_CONTRACT = "QR_DOJO_PROSPECTIVE_PAPER_PAIR_V1"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._@-]{0,95}$")
UTC = timezone.utc


class RoomRegistryError(ValueError):
    """The Paper room registry is unsafe or incomplete."""


def _aware_utc(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise RoomRegistryError(f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RoomRegistryError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC)


def load_room(registry_path: Path, room_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoomRegistryError(f"invalid room registry: {registry_path}") from exc
    if registry.get("contract") != REGISTRY_CONTRACT:
        raise RoomRegistryError("unsupported room registry contract")
    if not SAFE_ID.fullmatch(room_id):
        raise RoomRegistryError(f"unsafe room id: {room_id}")
    authority = registry.get("authority")
    if authority != {
        "paper_replay_only": True,
        "external_broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
        "automatic_deployment_allowed": False,
    }:
        raise RoomRegistryError("registry authority is invalid")
    rooms = registry.get("rooms")
    if not isinstance(rooms, list) or len(rooms) != 4:
        raise RoomRegistryError("prospective pair requires exactly four rooms")
    ids = [row.get("room_id") for row in rooms if isinstance(row, dict)]
    if len(ids) != len(rooms) or len(ids) != len(set(ids)):
        raise RoomRegistryError("registry room ids must be present and unique")
    try:
        room = next(row for row in rooms if row["room_id"] == room_id)
    except StopIteration as exc:
        raise RoomRegistryError(f"room is not registered: {room_id}") from exc
    return registry, room


def build_launch(
    *,
    registry_path: Path,
    room_id: str,
    python_executable: str,
    now_utc: datetime | None = None,
) -> tuple[list[str], dict[str, str], Path]:
    registry, room = load_room(registry_path, room_id)
    defaults = registry.get("defaults") or {}
    costs = room.get("costs") or {}
    config = room.get("bot_config")
    if not isinstance(config, dict):
        raise RoomRegistryError("room bot_config must be an object")
    if room.get("cost_arm") not in {"BASE", "STRESS"}:
        raise RoomRegistryError("room cost_arm must be BASE or STRESS")
    if room.get("management_arm") not in {"BOT_ONLY", "DIRECTION_GATE"}:
        raise RoomRegistryError("room management_arm is invalid")
    expected_policy = (
        "BOTH_SIDES"
        if room["management_arm"] == "BOT_ONLY"
        else "FOLLOW_24H_TREND"
    )
    if config.get("entry_direction_policy") != expected_policy:
        raise RoomRegistryError("room management arm and bot policy disagree")
    if config.get("signal") != "range_fade_limit":
        raise RoomRegistryError("prospective policy is bound to range_fade_limit")
    if {
        key: config.get(key)
        for key in (
            "external_broker_mutation_allowed",
            "live_permission",
            "order_authority",
        )
    } != {
        "external_broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
    }:
        raise RoomRegistryError("room bot authority is invalid")
    pairs = defaults.get("pairs")
    if not isinstance(pairs, list) or not pairs or len(pairs) != len(set(pairs)):
        raise RoomRegistryError("default pairs must be non-empty and unique")
    if sorted(config.get("pairs") or []) != sorted(pairs):
        raise RoomRegistryError("bot pairs must match registry pairs")
    experiment_id = str(registry.get("experiment_id") or "")
    if not SAFE_ID.fullmatch(experiment_id):
        raise RoomRegistryError("unsafe experiment id")
    launch_not_before = _aware_utc(
        registry.get("launch_not_before_utc"), "launch_not_before_utc"
    )
    window_end = _aware_utc(registry.get("window_end_utc"), "window_end_utc")
    now = (now_utc or datetime.now(UTC)).astimezone(UTC)
    if now < launch_not_before:
        raise RoomRegistryError("prospective window has not opened")
    if now >= window_end:
        raise RoomRegistryError("prospective window has already ended")
    minutes = (window_end - now).total_seconds() / 60.0
    if minutes <= 0:
        raise RoomRegistryError("prospective duration must be positive")

    session_dir = (
        REPO_ROOT / "research/data/dojo_paper_rooms_v1" / experiment_id / room_id
    )
    owner_id = str(config.get("strategy_owner_id") or "")
    if not owner_id or not SAFE_ID.fullmatch(owner_id):
        raise RoomRegistryError("strategy_owner_id is missing or unsafe")
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
        "--minutes",
        f"{minutes:.8f}",
        "--seed-oanda-m1-count",
        str(int(defaults["seed_oanda_m1_count"])),
        "--bot-module",
        str(REPO_ROOT / "bots/lab_bot.py") + ":Bot",
        "--strategy-owner-id",
        owner_id,
        "--bot-dependency",
        "src/quant_rabbit/dojo_bot_catalog.py",
        "--bot-dependency",
        "src/quant_rabbit/dojo_lab_provenance.py",
        "--bot-dependency",
        "src/quant_rabbit/virtual_broker.py",
        "--bot-dependency",
        str(registry_path.resolve().relative_to(REPO_ROOT)),
        "--bot-dependency",
        str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "--slippage-pips",
        str(float(costs["slippage_pips_per_fill"])),
        "--financing-pips-day",
        str(float(costs["financing_pips_per_day"])),
    ]
    env = dict(os.environ)
    env["DOJO_BOT_CONFIG"] = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.setdefault(
        "QR_OANDA_ENV_FILE", "/Users/tossaki/App/QuantRabbit-live/.env.local"
    )
    return command, env, session_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
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
    session_dir.mkdir(parents=True, exist_ok=False)
    os.execvpe(command[0], command, env)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
