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

from market_data import tick_window
from utils.market_hours import is_market_open
from utils.metrics_logger import log_metric
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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "on", "yes"}


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


def _latest_tick_lag_ms() -> Optional[float]:
    try:
        ticks = tick_window.recent_ticks(seconds=120.0, limit=1)
    except Exception:
        return None
    if not ticks:
        return None
    try:
        epoch = float(ticks[-1].get("epoch"))
    except (TypeError, ValueError, AttributeError):
        return None
    return max(0.0, (time.time() - epoch) * 1000.0)


def _emit_slo_metrics(loop_start_mono: float) -> tuple[Optional[float], float] | None:
    if not is_market_open():
        return None
    data_lag_ms = _latest_tick_lag_ms()
    if data_lag_ms is not None:
        log_metric("data_lag_ms", data_lag_ms, tags={"mode": "strategy_control"})
    decision_latency_ms = max(0.0, (time.monotonic() - loop_start_mono) * 1000.0)
    log_metric(
        "decision_latency_ms",
        decision_latency_ms,
        tags={"mode": "strategy_control"},
    )
    return data_lag_ms, decision_latency_ms


async def strategy_control_worker() -> None:
    poll_interval_sec = max(1.0, _env_float("STRATEGY_CONTROL_POLL_SEC", 5.0))
    heartbeat_sec = max(30.0, _env_float("STRATEGY_CONTROL_HEARTBEAT_SEC", 60.0))
    slo_metrics_enabled = _env_bool("STRATEGY_CONTROL_SLO_METRICS_ENABLED", True)
    slo_metrics_interval_sec = max(
        poll_interval_sec,
        _env_float("STRATEGY_CONTROL_SLO_METRICS_INTERVAL_SEC", 10.0),
    )
    command_path = _env_str("STRATEGY_CONTROL_COMMAND_PATH", "")
    command_mtime: Optional[float] = None

    strategy_control.sync_env_overrides()

    last_heartbeat = time.monotonic() - heartbeat_sec
    last_slo_metrics = time.monotonic() - slo_metrics_interval_sec
    while True:
        loop_start = time.monotonic()
        now = loop_start
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
            if slo_metrics_enabled and now - last_slo_metrics >= slo_metrics_interval_sec:
                emitted = _emit_slo_metrics(loop_start)
                if emitted is not None:
                    data_lag_ms, decision_latency_ms = emitted
                    LOG.debug(
                        "[STRATEGY_CONTROL] slo_metrics data_lag_ms=%s decision_latency_ms=%.2f",
                        f"{data_lag_ms:.1f}" if data_lag_ms is not None else "na",
                        decision_latency_ms,
                    )
                last_slo_metrics = now
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
