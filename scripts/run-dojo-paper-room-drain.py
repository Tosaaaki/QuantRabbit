#!/usr/bin/env python3
"""Drain one stopped formal DOJO paper room without admitting new entries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _room_launcher_module():
    path = REPO_ROOT / "scripts/run-dojo-paper-room.py"
    spec = importlib.util.spec_from_file_location("dojo_paper_room_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load room launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_drain_launch(
    *, registry_path: Path, room_id: str, python_executable: str
) -> tuple[list[str], dict[str, str], Path]:
    launcher = _room_launcher_module()
    registry, room = launcher.load_room(registry_path, room_id)
    defaults = registry.get("defaults") or {}
    costs = room.get("costs") or {}
    config = room.get("bot_config") or {}
    ceiling = config.get("ceiling_min")
    if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling <= 0:
        raise launcher.RoomRegistryError("room drain requires a positive ceiling_min")
    pairs = defaults.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise launcher.RoomRegistryError("room drain requires registered pairs")
    _, _, session_dir = launcher.build_launch(
        registry_path=registry_path,
        room_id=room_id,
        python_executable=python_executable,
    )
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
        "1440",
        "--drain-only",
        "--drain-ceiling-min",
        str(ceiling),
        "--slippage-pips",
        str(float(costs["slippage_pips_per_fill"])),
        "--financing-pips-day",
        str(float(costs["financing_pips_per_day"])),
        "--leverage",
        str(float(defaults["leverage"])),
        "--runtime-dependency",
        str(registry_path.resolve()),
        "--runtime-dependency",
        str(Path(__file__).resolve()),
    ]
    env = dict(os.environ)
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
    command, env, session_dir = build_drain_launch(
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
                },
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
