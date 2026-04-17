#!/usr/bin/env python3
"""
Brake Gate — small helper for entry bots to honor inventory_brake + regime_switch.

Read by range_bot / trend_bot at entry-decision time:
    blocked, reason = brake_gate.check(pair, direction)
    if blocked:
        skip
        log reason

Direction can be "BUY"/"SELL" or "LONG"/"SHORT" — both accepted.

State files:
    logs/bot_brake_state.json     (inventory_brake.py)
    logs/bot_regime_state.json    (regime_switch.py)

If a state file is stale (> STATE_MAX_AGE_SEC) or missing, the gate is permissive
(returns no-block). That keeps the entry bot from freezing if the brake loop dies.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_MAIN_ROOT = ROOT
if not (_MAIN_ROOT / "config" / "env.toml").exists():
    try:
        _git_common = Path(subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(ROOT), text=True
        ).strip())
        _MAIN_ROOT = _git_common.resolve().parent
    except Exception:
        pass

BRAKE_STATE_PATH = _MAIN_ROOT / "logs" / "bot_brake_state.json"
REGIME_STATE_PATH = _MAIN_ROOT / "logs" / "bot_regime_state.json"

STATE_MAX_AGE_SEC = 300  # 5 min — anything older = ignore (loop probably dead)


def _normalize_dir(d: str) -> str:
    d = (d or "").upper()
    if d in ("BUY", "LONG"):
        return "LONG"
    if d in ("SELL", "SHORT"):
        return "SHORT"
    return d


def _read_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    ts = data.get("ts")
    if not ts:
        return data
    try:
        t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - t).total_seconds()
        if age > STATE_MAX_AGE_SEC:
            return None
    except Exception:
        return data
    return data


def load_brake_state() -> dict | None:
    return _read_state(BRAKE_STATE_PATH)


def load_regime_state() -> dict | None:
    return _read_state(REGIME_STATE_PATH)


def check(pair: str, direction: str, lane: str = "default") -> tuple[bool, str]:
    """Return (blocked, reason). blocked=True means the entry bot must SKIP this entry.

    lane="range_scalp": mean-reversion, tight structural SL, small size.
      - Ignores global_halt at CAUTION (only PANIC stage halts).
      - Still respects pair-level block and regime TREND block.

    Checked in order:
      1. brake.global_halt_new — margin >= CAUTION (range_scalp skips unless PANIC)
      2. brake.pair[pair].block_{long|short}_new — imbalance + heavy stranded side
      3. regime.pair[pair].regime == TREND and direction != trend_dir — counter-trend block
    """
    side = _normalize_dir(direction)
    brake = load_brake_state()
    regime = load_regime_state()

    if brake:
        if brake.get("global_halt_new"):
            stage = brake.get("margin_stage", "?")
            if lane == "range_scalp" and stage != "PANIC":
                pass  # range scalp harvests chop at CAUTION — only PANIC halts it
            else:
                return True, f"brake_global_halt margin_stage={stage}"
        pair_b = (brake.get("pairs") or {}).get(pair) or {}
        if side == "LONG" and pair_b.get("block_long_new"):
            return True, f"brake_pair {pair_b.get('reason','imbalance')}"
        if side == "SHORT" and pair_b.get("block_short_new"):
            return True, f"brake_pair {pair_b.get('reason','imbalance')}"

    if regime:
        pair_r = (regime.get("pairs") or {}).get(pair) or {}
        if pair_r.get("regime") == "TREND":
            td = pair_r.get("trend_dir")
            if td and side and side != td:
                return True, f"regime_trend td={td} adx_m5={pair_r.get('adx_m5')}"

    return False, ""


def is_drain_mode(pair: str) -> bool:
    brake = load_brake_state()
    if not brake:
        return False
    pair_b = (brake.get("pairs") or {}).get(pair) or {}
    return bool(pair_b.get("drain_mode"))


def margin_stage() -> str:
    brake = load_brake_state()
    if not brake:
        return "UNKNOWN"
    return brake.get("margin_stage", "UNKNOWN")
