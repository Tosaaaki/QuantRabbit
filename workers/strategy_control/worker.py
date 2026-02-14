"""Strategy control worker.

Periodically applies environment/command based flags to
`workers.common.strategy_control`, so strategies can query enable/disable
state at runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from workers.common import strategy_control


LOG = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _snapshot() -> tuple[bool, bool, bool, int]:
    entry, exit_enabled, lock = strategy_control.get_global_flags()
    return bool(entry), bool(exit_enabled), bool(lock), len(strategy_control.list_enabled_strategies())


def _sync_from_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        LOG.debug("[STRATEGY_CONTROL] command file read failed: %s", exc)
        return

    try:
        payload = json.loads(text)
    except Exception as exc:
        LOG.warning("[STRATEGY_CONTROL] invalid JSON in %s: %s", path, exc)
        return

    if not isinstance(payload, dict):
        LOG.warning("[STRATEGY_CONTROL] invalid payload type in %s: %s", path, type(payload).__name__)
        return

    global_cfg = payload.get("global", {})
    if isinstance(global_cfg, dict):
        strategy_control.set_global_flags(
            entry=global_cfg.get("entry_enabled"),
            exit=global_cfg.get("exit_enabled"),
            lock=global_cfg.get("global_lock"),
            note="command_file",
        )

    for slug, raw in (payload.get("strategies") or {}).items():
        if not isinstance(raw, dict):
            continue
        if "global_lock" in raw:
            lock_value = raw.get("global_lock")
        else:
            lock_value = raw.get("lock")
        strategy_control.set_strategy_flags(
            str(slug),
            entry=raw.get("entry_enabled"),
            exit=raw.get("exit_enabled"),
            lock=lock_value,
            note="command_file",
        )


async def strategy_control_worker() -> None:
    poll_interval_sec = max(1.0, _env_float("STRATEGY_CONTROL_POLL_SEC", 5.0))
    heartbeat_sec = max(30.0, _env_float("STRATEGY_CONTROL_HEARTBEAT_SEC", 60.0))
    command_path = _env_str("STRATEGY_CONTROL_COMMAND_PATH", "")
    command_mtime: Optional[float] = None

    if command_path:
        strategy_control.sync_env_overrides()

    last_heartbeat = time.monotonic() - heartbeat_sec
    while True:
        now = time.monotonic()
        try:
            strategy_control.sync_env_overrides()

            if command_path:
                file_path = Path(command_path)
                try:
                    stat = file_path.stat()
                    stat_mtime = stat.st_mtime
                except FileNotFoundError:
                    stat_mtime = None
                if stat_mtime is not None and stat_mtime != command_mtime:
                    command_mtime = stat_mtime
                    _sync_from_file(file_path)

            if now - last_heartbeat >= heartbeat_sec:
                entry_global, exit_global, lock, total = _snapshot()
                LOG.info(
                    "[STRATEGY_CONTROL] heartbeat global(entry=%s, exit=%s, lock=%s) strategies=%d",
                    entry_global,
                    exit_global,
                    lock,
                    total,
                )
                last_heartbeat = now
        except Exception as exc:
            LOG.warning("[STRATEGY_CONTROL] sync loop error: %s", exc)

        await asyncio.sleep(poll_interval_sec)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


if __name__ == "__main__":
    _configure_logging()
    asyncio.run(strategy_control_worker())
