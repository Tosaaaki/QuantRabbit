#!/usr/bin/env python3
"""Launch one independent archived-worker A/B Paper room."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.dojo_legacy_worker_comparison import AUTHORITY


ROOT = Path(__file__).resolve().parents[1]
SAFE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry.get("contract") != "QR_DOJO_LEGACY_WORKER_PAPER_ROOMS_V1" or registry.get("authority") != AUTHORITY:
        raise SystemExit("invalid Paper registry contract/authority")
    if not SAFE.fullmatch(args.room_id):
        raise SystemExit("unsafe room id")
    room = next((row for row in registry.get("rooms", []) if row.get("room_id") == args.room_id), None)
    if room is None:
        raise SystemExit("room is not registered")
    config = dict(room["bot_config"])
    if config.get("authority") != AUTHORITY:
        raise SystemExit("room authority invalid")
    if room["management_arm"] != config.get("management_arm"):
        raise SystemExit("room arm/config mismatch")
    if room["family"] != config.get("family"):
        raise SystemExit("room family/config mismatch")
    end = datetime.fromisoformat(str(registry["window_end_utc"]).replace("Z", "+00:00")).astimezone(timezone.utc)
    minutes = (end - datetime.now(timezone.utc)).total_seconds() / 60.0
    if minutes <= 0:
        raise SystemExit("Paper window ended")
    experiment = str(registry["experiment_id"])
    if not SAFE.fullmatch(experiment):
        raise SystemExit("unsafe experiment id")
    session_dir = ROOT / "research/data/dojo_paper_rooms_v1" / experiment / args.room_id
    command = [
        sys.executable,
        str(ROOT / "scripts/run-virtual-market-session.py"),
        "--feed", "live",
        "--session-dir", str(session_dir),
        "--pairs", "USD_JPY",
        "--balance", str(float(registry["initial_balance_jpy"])),
        "--minutes", f"{minutes:.8f}",
        "--seed-oanda-m1-count", "1500",
        "--bot-module", str(ROOT / "bots/legacy_worker_paper_bot.py") + ":Bot",
        "--strategy-owner-id", str(config["strategy_owner_id"]),
        "--bot-dependency", "bots/legacy_worker_paper_bot.py",
        "--bot-dependency", "src/quant_rabbit/dojo_legacy_worker_comparison.py",
        "--bot-dependency", "src/quant_rabbit/dojo_lab_provenance.py",
        "--bot-dependency", "src/quant_rabbit/virtual_broker.py",
        "--bot-dependency", "config/dojo_legacy_ai_inventory_policy_v1.json",
        "--bot-dependency", str(args.registry.resolve().relative_to(ROOT)),
        "--bot-dependency", "scripts/run-dojo-legacy-paper-room.py",
        "--slippage-pips", str(float(registry["slippage_pips_per_fill"])),
        "--financing-pips-day", str(float(registry["financing_pips_per_day"])),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["DOJO_BOT_CONFIG"] = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    env.setdefault("QR_OANDA_ENV_FILE", "/Users/tossaki/App/QuantRabbit-live/.env.local")
    if room["management_arm"] == "AI_INVENTORY":
        env["DOJO_AI_DECISION_LEDGER"] = str(session_dir / "ai_decisions.jsonl")
    if args.dry_run:
        print(json.dumps({"room_id": args.room_id, "session_dir": str(session_dir), "command": command, "bot_config": config}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    session_dir.mkdir(parents=True, exist_ok=False)
    os.execvpe(command[0], command, env)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
