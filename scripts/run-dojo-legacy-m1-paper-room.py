#!/usr/bin/env python3
"""Launch one create-once M1Scalper Paper room while holding an OS lock."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.dojo_legacy_worker_comparison import AUTHORITY


ROOT = Path(__file__).resolve().parents[1]
SAFE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
CONTRACT = "QR_DOJO_LEGACY_M1_PAPER_ROOMS_V1"


def _load(registry_path: Path, room_id: str) -> tuple[dict, dict]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("contract") != CONTRACT or registry.get("authority") != AUTHORITY:
        raise SystemExit("invalid M1 Paper registry contract/authority")
    if not SAFE.fullmatch(room_id):
        raise SystemExit("unsafe room id")
    rooms = list(registry.get("rooms") or [])
    ids = [row.get("room_id") for row in rooms]
    operations = [(row.get("bot_config") or {}).get("operation_id") for row in rooms]
    owners = [(row.get("bot_config") or {}).get("strategy_owner_id") for row in rooms]
    if (
        len(ids) != len(set(ids))
        or len(operations) != len(set(operations))
        or len(owners) != len(set(owners))
    ):
        raise SystemExit("duplicate room, operation_id, or owner is forbidden")
    room = next((row for row in rooms if row.get("room_id") == room_id), None)
    if room is None:
        raise SystemExit("room is not registered")
    config = dict(room.get("bot_config") or {})
    if (
        config.get("authority") != AUTHORITY
        or config.get("management_arm") != room.get("management_arm")
        or int(config.get("fixed_units") or 0) > 1000
    ):
        raise SystemExit("room config is unsafe")
    return registry, room


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registry, room = _load(args.registry.resolve(), args.room_id)
    experiment = str(registry["experiment_id"])
    if not SAFE.fullmatch(experiment):
        raise SystemExit("unsafe experiment id")
    end = datetime.fromisoformat(
        str(registry["window_end_utc"]).replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    minutes = (end - datetime.now(timezone.utc)).total_seconds() / 60.0
    if minutes <= 0:
        raise SystemExit("M1 Paper window ended")
    session_dir = (
        ROOT
        / "research"
        / "data"
        / "dojo_paper_rooms_v1"
        / experiment
        / args.room_id
    )
    config = dict(room["bot_config"])
    cost = dict(registry["cost_model"])
    command = [
        sys.executable,
        str(ROOT / "scripts/run-virtual-market-session.py"),
        "--feed",
        "live",
        "--session-dir",
        str(session_dir),
        "--pairs",
        "USD_JPY",
        "--balance",
        str(float(registry["initial_balance_jpy"])),
        "--minutes",
        f"{minutes:.8f}",
        "--seed-oanda-m1-count",
        "1500",
        "--bot-module",
        str(ROOT / "bots/legacy_m1_scalper_paper_bot.py") + ":Bot",
        "--strategy-owner-id",
        str(config["strategy_owner_id"]),
        "--bot-dependency",
        "bots/legacy_m1_scalper_paper_bot.py",
        "--bot-dependency",
        "src/quant_rabbit/dojo_legacy_m1_signal.py",
        "--bot-dependency",
        "src/quant_rabbit/legacy_m1_frozen/__init__.py",
        "--bot-dependency",
        "src/quant_rabbit/legacy_m1_frozen/m1_scalper_d8f751afc.py",
        "--bot-dependency",
        "src/quant_rabbit/legacy_m1_frozen/calc_core_d8f751afc.py",
        "--bot-dependency",
        "config/dojo_legacy_m1_scalper_d8f751afc.json",
        "--bot-dependency",
        "config/dojo_legacy_m1_ai_inventory_policy_v1.json",
        "--bot-dependency",
        str(args.registry.resolve().relative_to(ROOT)),
        "--bot-dependency",
        "scripts/run-dojo-legacy-m1-paper-room.py",
        "--bot-dependency",
        "src/quant_rabbit/dojo_legacy_worker_comparison.py",
        "--bot-dependency",
        "src/quant_rabbit/dojo_lab_provenance.py",
        "--bot-dependency",
        "src/quant_rabbit/virtual_broker.py",
        "--slippage-pips",
        str(float(cost["slippage_pips_per_fill"])),
        "--financing-pips-day",
        str(float(cost["financing_pips_per_day"])),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["DOJO_BOT_CONFIG"] = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    env["DOJO_M1_DECISION_LEDGER"] = str(session_dir / "decisions.jsonl")
    env.setdefault(
        "QR_OANDA_ENV_FILE", "/Users/tossaki/App/QuantRabbit-live/.env.local"
    )
    lock_path = (
        ROOT
        / "research"
        / "data"
        / "dojo_paper_rooms_v1"
        / ".locks"
        / experiment
        / f"{args.room_id}.lock"
    )
    summary = {
        "room_id": args.room_id,
        "operation_id": config["operation_id"],
        "session_dir": str(session_dir),
        "lock_path": str(lock_path),
        "command": command,
        "authority": AUTHORITY,
        "fixed_units": config["fixed_units"],
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("M1 Paper room is already running") from exc
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "operation_id": config["operation_id"],
                    "room_id": args.room_id,
                },
                sort_keys=True,
            )
            + "\n"
        )
        lock_handle.flush()
        os.fsync(lock_handle.fileno())
        session_dir.mkdir(parents=True, exist_ok=False)
        completed = subprocess.run(command, env=env, check=False)
        return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
