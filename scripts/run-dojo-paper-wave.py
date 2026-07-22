#!/usr/bin/env python3
"""Start every room in one pre-registered DOJO paper comparison wave."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


def _room_launcher_module():
    path = REPO_ROOT / "scripts/run-dojo-paper-room.py"
    spec = importlib.util.spec_from_file_location("dojo_paper_room_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load room launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_wave(registry_path: Path, python_executable: str) -> list[dict]:
    launcher = _room_launcher_module()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = []
    for room in registry.get("rooms") or []:
        room_id = room.get("room_id")
        launcher.load_room(registry_path, room_id)
        command = [
            python_executable,
            str(REPO_ROOT / "scripts/run-dojo-paper-room.py"),
            "--registry",
            str(registry_path),
            "--room-id",
            room_id,
        ]
        _, _, session_dir = launcher.build_launch(
            registry_path=registry_path,
            room_id=room_id,
            python_executable=python_executable,
        )
        rows.append(
            {
                "room_id": room_id,
                "screen_name": f"qr-dojo-{room_id}",
                "session_dir": session_dir,
                "command": command,
            }
        )
    return rows


def _assert_launch_window(registry_path: Path, lead_seconds: int) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    window = registry.get("window") or {}
    start = datetime.fromisoformat(str(window.get("start_utc")))
    end = datetime.fromisoformat(str(window.get("end_utc")))
    now = datetime.now(UTC)
    if now.timestamp() < start.timestamp() - lead_seconds:
        raise RuntimeError("paper wave launch is earlier than the allowed lead time")
    if now.timestamp() > start.timestamp() + 60:
        raise RuntimeError("fresh paper wave launch is over 60 seconds late")
    if now >= end:
        raise RuntimeError("paper wave window has ended")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "config/dojo_paper_rooms_v1.json",
    )
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--lead-seconds", type=int, default=600)
    args = parser.parse_args()
    registry_path = args.registry.resolve()
    rows = build_wave(registry_path, sys.executable)
    if not args.start:
        print(
            json.dumps(
                [
                    {
                        **row,
                        "session_dir": str(row["session_dir"]),
                    }
                    for row in rows
                ],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    _assert_launch_window(registry_path, args.lead_seconds)
    active = subprocess.run(
        ["screen", "-ls"], capture_output=True, text=True, check=False
    ).stdout
    started = []
    for row in rows:
        name = row["screen_name"]
        if f".{name}\t" in active:
            continue
        session_dir = row["session_dir"]
        session_dir.mkdir(parents=True, exist_ok=True)
        log_path = session_dir / "screen.log"
        subprocess.run(
            [
                "screen",
                "-L",
                "-Logfile",
                str(log_path),
                "-dmS",
                name,
                *row["command"],
            ],
            check=True,
        )
        started.append(name)
    print(json.dumps({"started": started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
