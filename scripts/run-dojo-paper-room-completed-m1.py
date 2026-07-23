#!/usr/bin/env python3
"""Launch one future diagnostic room on completed M1 decision evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE = "OANDA_COMPLETED_M1_BID_ASK_V1"


def _base_launcher() -> Any:
    path = REPO_ROOT / "scripts/run-dojo-paper-room.py"
    spec = importlib.util.spec_from_file_location(
        "dojo_base_paper_room_launcher",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base room launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_launch(
    *,
    registry_path: Path,
    room_id: str,
    python_executable: str,
) -> tuple[list[str], dict[str, str], Path]:
    launcher = _base_launcher()
    registry, _room = launcher.load_room(registry_path, room_id)
    defaults = registry.get("defaults") or {}
    if registry.get("proof_mode") != "diagnostic":
        raise launcher.RoomRegistryError(
            "completed-M1 measurement repair is diagnostic-only"
        )
    if defaults.get("live_bot_bar_source") != EXPECTED_SOURCE:
        raise launcher.RoomRegistryError(
            "registry does not declare the completed-M1 decision source"
        )
    bot_module = str(defaults.get("bot_module") or "")
    if bot_module.partition(":")[0] != "bots/lab_bot_observed.py":
        raise launcher.RoomRegistryError(
            "completed-M1 diagnostic requires the observed lab bot"
        )

    command, env, session_dir = launcher.build_launch(
        registry_path=registry_path,
        room_id=room_id,
        python_executable=python_executable,
    )
    command[1] = str(
        REPO_ROOT / "scripts/run-virtual-market-session-completed-m1.py"
    )
    seed_root = REPO_ROOT / str(defaults.get("seed_m1_root") or "")
    seed_paths = sorted(
        path
        for pair in defaults.get("pairs") or []
        for path in seed_root.glob(f"*/{pair}/{pair}_M1_BA_*.jsonl.gz")
    )
    if not seed_paths:
        raise launcher.RoomRegistryError(
            "completed-M1 diagnostic seed shards are absent"
        )
    dependencies = (
        REPO_ROOT / "scripts/run-virtual-market-session.py",
        REPO_ROOT / "scripts/run-dojo-paper-room-completed-m1.py",
        REPO_ROOT / "src/quant_rabbit/dojo_completed_m1_live.py",
        REPO_ROOT / "bots/lab_bot.py",
        *seed_paths,
    )
    for path in dependencies:
        command.extend(["--runtime-dependency", str(path)])
    env["QR_DOJO_LIVE_BOT_BAR_SOURCE"] = EXPECTED_SOURCE
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
                    "live_bot_bar_source": env[
                        "QR_DOJO_LIVE_BOT_BAR_SOURCE"
                    ],
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
