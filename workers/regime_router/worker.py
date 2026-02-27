"""Regime router worker.

Routes strategy entry enablement by current macro/micro regime and writes
per-strategy flags into `workers.common.strategy_control`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from workers.common import strategy_control
from workers.common.quality_gate import current_regime


LOG = logging.getLogger(__name__)

_FALSE_VALUES = {"", "0", "false", "no", "off"}
_VALID_REGIMES = {"trend", "range", "breakout", "mixed", "event"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in _FALSE_VALUES


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


def _normalize_strategy_list(raw: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for token in str(raw or "").replace("\n", ",").split(","):
        slug = strategy_control.normalize_strategy_slug(token)
        if not slug or slug in seen:
            continue
        out.append(slug)
        seen.add(slug)
    return tuple(out)


def _normalize_regime(value: Optional[str]) -> str:
    text = str(value or "").strip().lower()
    if text in _VALID_REGIMES:
        return text
    return ""


def _decide_candidate_route(macro_regime: Optional[str], micro_regime: Optional[str]) -> str:
    macro = _normalize_regime(macro_regime)
    micro = _normalize_regime(micro_regime)

    if not macro and not micro:
        return "unknown"
    if macro == "event" or micro == "event":
        return "event"
    if micro == "trend" and macro in {"trend", "breakout"}:
        return "trend"
    if micro == "breakout":
        return "breakout"
    if micro == "range" and macro in {"range", "mixed"}:
        return "range"
    return "mixed"


def _apply_dwell(
    *,
    active_route: str,
    active_since_mono: float,
    candidate_route: str,
    now_mono: float,
    min_dwell_sec: float,
) -> tuple[str, float, bool]:
    if not active_route:
        return candidate_route, now_mono, True
    if candidate_route == active_route:
        return active_route, active_since_mono, False
    if now_mono - active_since_mono < max(0.0, float(min_dwell_sec)):
        return active_route, active_since_mono, False
    return candidate_route, now_mono, True


def _build_entry_plan(managed_strategies: tuple[str, ...], enabled_strategies: set[str]) -> dict[str, bool]:
    return {slug: (slug in enabled_strategies) for slug in managed_strategies}


def _apply_entry_plan(entry_plan: dict[str, bool], *, note: str) -> tuple[int, int]:
    changed = 0
    for slug, desired_entry in entry_plan.items():
        current = strategy_control.get_flags(slug)
        # Missing rows default to entry enabled in strategy_control.
        current_entry = True if current is None else bool(current[0])
        if current_entry == bool(desired_entry):
            continue
        strategy_control.set_strategy_flags(slug, entry=bool(desired_entry), note=note[:255])
        changed += 1
    return changed, len(entry_plan)


def _write_snapshot(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        temp_path = Path(fh.name)
    temp_path.replace(path)


@dataclass(frozen=True)
class RouterConfig:
    enabled: bool
    poll_sec: float
    heartbeat_sec: float
    min_dwell_sec: float
    macro_tf: str
    micro_tf: str
    event_mode: bool
    snapshot_path: Path
    managed_strategies: tuple[str, ...]
    route_targets: dict[str, set[str]]


def _load_config() -> RouterConfig:
    default_managed = "scalp_ping_5s_b,scalp_ping_5s_c,scalp_ping_5s_d,scalp_ping_5s_flow"
    managed = _normalize_strategy_list(
        _env_str("REGIME_ROUTER_MANAGED_STRATEGIES", default_managed)
    )
    route_targets = {
        "trend": set(
            _normalize_strategy_list(
                _env_str("REGIME_ROUTER_TREND_ENTRY_STRATEGIES", "scalp_ping_5s_d")
            )
        ),
        "breakout": set(
            _normalize_strategy_list(
                _env_str("REGIME_ROUTER_BREAKOUT_ENTRY_STRATEGIES", "scalp_ping_5s_d")
            )
        ),
        "range": set(
            _normalize_strategy_list(
                _env_str("REGIME_ROUTER_RANGE_ENTRY_STRATEGIES", "scalp_ping_5s_c")
            )
        ),
        "mixed": set(
            _normalize_strategy_list(
                _env_str("REGIME_ROUTER_MIXED_ENTRY_STRATEGIES", "scalp_ping_5s_c")
            )
        ),
        "event": set(
            _normalize_strategy_list(_env_str("REGIME_ROUTER_EVENT_ENTRY_STRATEGIES", ""))
        ),
        "unknown": set(
            _normalize_strategy_list(
                _env_str("REGIME_ROUTER_UNKNOWN_ENTRY_STRATEGIES", "scalp_ping_5s_c")
            )
        ),
    }
    if not managed:
        inferred: list[str] = []
        seen: set[str] = set()
        for targets in route_targets.values():
            for slug in sorted(targets):
                if slug in seen:
                    continue
                seen.add(slug)
                inferred.append(slug)
        managed = tuple(inferred)

    snapshot_path = Path(
        _env_str("REGIME_ROUTER_SNAPSHOT_PATH", "logs/regime_router_state.json")
    )
    if not snapshot_path.is_absolute():
        snapshot_path = (Path.cwd() / snapshot_path).resolve()

    return RouterConfig(
        enabled=_env_bool("REGIME_ROUTER_ENABLED", False),
        poll_sec=max(1.0, _env_float("REGIME_ROUTER_POLL_SEC", 5.0)),
        heartbeat_sec=max(10.0, _env_float("REGIME_ROUTER_HEARTBEAT_SEC", 60.0)),
        min_dwell_sec=max(0.0, _env_float("REGIME_ROUTER_MIN_DWELL_SEC", 30.0)),
        macro_tf=_env_str("REGIME_ROUTER_MACRO_TF", "H4").upper() or "H4",
        micro_tf=_env_str("REGIME_ROUTER_MICRO_TF", "M1").upper() or "M1",
        event_mode=_env_bool("REGIME_ROUTER_EVENT_MODE", False),
        snapshot_path=snapshot_path,
        managed_strategies=managed,
        route_targets=route_targets,
    )


async def regime_router_worker() -> None:
    cfg = _load_config()
    if not cfg.enabled:
        LOG.info("[REGIME_ROUTER] disabled by REGIME_ROUTER_ENABLED=0")
        while True:
            await asyncio.sleep(3600.0)

    active_route = ""
    active_since_mono = 0.0
    last_heartbeat = time.monotonic() - cfg.heartbeat_sec

    LOG.info(
        "[REGIME_ROUTER] start managed=%s macro_tf=%s micro_tf=%s poll=%.1fs dwell=%.1fs",
        ",".join(cfg.managed_strategies),
        cfg.macro_tf,
        cfg.micro_tf,
        cfg.poll_sec,
        cfg.min_dwell_sec,
    )

    while True:
        now_mono = time.monotonic()
        macro_regime = current_regime(cfg.macro_tf, event_mode=cfg.event_mode)
        micro_regime = current_regime(cfg.micro_tf, event_mode=cfg.event_mode)
        candidate_route = _decide_candidate_route(macro_regime, micro_regime)

        active_route, active_since_mono, switched = _apply_dwell(
            active_route=active_route,
            active_since_mono=active_since_mono,
            candidate_route=candidate_route,
            now_mono=now_mono,
            min_dwell_sec=cfg.min_dwell_sec,
        )

        enabled_targets = cfg.route_targets.get(active_route) or cfg.route_targets.get("unknown", set())
        entry_plan = _build_entry_plan(cfg.managed_strategies, enabled_targets)
        note = (
            f"regime_router:{active_route}|macro={macro_regime or 'na'}|micro={micro_regime or 'na'}"
        )
        changed, total = _apply_entry_plan(entry_plan, note=note)

        state = {
            "updated_at_epoch": time.time(),
            "active_route": active_route,
            "candidate_route": candidate_route,
            "macro_regime": macro_regime or "",
            "micro_regime": micro_regime or "",
            "switched": bool(switched),
            "managed_strategies": list(cfg.managed_strategies),
            "enabled_strategies": sorted(enabled_targets),
            "entry_plan": entry_plan,
            "changed_count": int(changed),
            "total_count": int(total),
            "event_mode": bool(cfg.event_mode),
            "macro_tf": cfg.macro_tf,
            "micro_tf": cfg.micro_tf,
        }
        try:
            _write_snapshot(cfg.snapshot_path, state)
        except Exception as exc:
            LOG.warning("[REGIME_ROUTER] snapshot write failed: %s", exc)

        if switched or changed > 0 or (now_mono - last_heartbeat >= cfg.heartbeat_sec):
            LOG.info(
                "[REGIME_ROUTER] route=%s candidate=%s macro=%s micro=%s enabled=%s changed=%d/%d",
                active_route,
                candidate_route,
                macro_regime or "na",
                micro_regime or "na",
                ",".join(sorted(enabled_targets)) or "-",
                changed,
                total,
            )
            last_heartbeat = now_mono

        await asyncio.sleep(cfg.poll_sec)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


if __name__ == "__main__":
    _configure_logging()
    asyncio.run(regime_router_worker())

