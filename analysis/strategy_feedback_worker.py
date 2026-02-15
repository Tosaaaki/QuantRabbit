#!/usr/bin/env python3
"""Generate per-strategy analysis feedback JSON for runtime entry tuning.

This worker discovers active strategy workers dynamically and calculates lightweight
performance metrics from `logs/trades.db`, then writes knobs consumed by
`analysis.strategy_feedback.current_advice()`.

Design goals:
- strategy discovery follows strategy workers/services/strategy_control/trade history
- added or removed strategy workers are reflected automatically
- each strategy uses a dedicated "analysis squad" (scalp/micro/macro/session)
  with a common baseline formula
- generate best-effort knobs: entry probability, units, and TP/SL distance
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent


class WorkerConfig:
    def __init__(self) -> None:
        self.trades_db = Path(os.getenv("STRATEGY_FEEDBACK_TRADES_DB", BASE_DIR / "logs" / "trades.db"))
        raw_feedback_path = os.getenv(
            "STRATEGY_FEEDBACK_PATH", BASE_DIR / "logs" / "strategy_feedback.json"
        )
        self.feedback_path = Path(raw_feedback_path)
        self.systemd_dir = Path(os.getenv("STRATEGY_FEEDBACK_SYSTEMD_DIR", BASE_DIR / "systemd"))
        self.lookback_days = int(os.getenv("STRATEGY_FEEDBACK_LOOKBACK_DAYS", "14"))
        self.min_trades = int(os.getenv("STRATEGY_FEEDBACK_MIN_TRADES", "12"))
        self.keep_inactive_days = int(os.getenv("STRATEGY_FEEDBACK_KEEP_INACTIVE_DAYS", "14"))
        self.force = False


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _norm_tag(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^0-9a-z_]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


@dataclass
class StrategyRecord:
    canonical_tag: str
    sources: set[str] = field(default_factory=set)
    active: bool = False
    entry_active: bool = False
    exit_active: bool = False
    enabled: bool | None = None
    last_closed: str | None = None


@dataclass
class StrategyStats:
    tag: str
    trades: int
    wins: int
    losses: int
    sum_pips: float
    avg_pips: float
    avg_abs_pips: float
    gross_win: float
    gross_loss: float
    avg_hold_sec: float | None
    last_closed: str | None

    @property
    def win_rate(self) -> float:
        if self.trades <= 0:
            return 0.0
        return self.wins / self.trades

    @property
    def avg_win(self) -> float:
        return _to_float(self.gross_win / self.wins, 0.0)

    @property
    def avg_loss(self) -> float:
        return _to_float(self.gross_loss / self.losses, 0.0)

    @property
    def profit_factor(self) -> float:
        if self.gross_loss <= 0:
            return float("inf") if self.wins > 0 else 0.0
        return abs(self.gross_win / self.gross_loss)

    @property
    def loss_asymmetry(self) -> float:
        if self.avg_loss <= 0:
            return 0.0
        if self.avg_win <= 0:
            return float("inf")
        return self.avg_loss / self.avg_win


def _systemctl_available() -> bool:
    return shutil.which("systemctl") is not None


def _systemctl_running_services() -> set[str]:
    services: set[str] = set()
    if not _systemctl_available():
        return services
    try:
        cp = subprocess.run(
            ["systemctl", "list-units", "--type=service", "--state=running", "--no-pager", "--no-legend"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except Exception:
        return services
    if cp.returncode != 0:
        return services
    for line in cp.stdout.splitlines():
        if not line.strip():
            continue
        name = line.split(None, 1)[0].strip()
        if name:
            services.add(name)
    return services


def _read_service_units(systemd_dir: Path) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    for path in sorted(systemd_dir.glob("quant-*.service")):
        try:
            units.append((path.name, path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return units


def _parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if key:
                data[key.strip()] = value
    except Exception:
        return {}
    return data


def _split_csv(raw: str) -> list[str]:
    out: list[str] = []
    for token in re.split(r"[;,\n\r]+", raw):
        token = token.strip()
        if token:
            out.append(token)
    return out


def _looks_like_tag_value(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    if token.lower() in {"0", "1", "true", "false", "on", "off", "yes", "no"}:
        return False
    if re.search(r"[/\\]", token):
        return False
    return True


def _extract_tag_candidates_from_env(env: dict[str, str]) -> set[str]:
    tags: set[str] = set()
    for key, value in env.items():
        key_u = str(key or "").upper()
        if not _looks_like_tag_value(str(value)):
            continue
        values = _split_csv(str(value))
        if not values:
            continue

        is_mode = key_u.endswith("_MODE")
        is_tags = key_u.endswith("_TAGS")
        is_strategy_tag = key_u.endswith("_STRATEGY_TAG") or key_u.endswith("_STRATEGY_TAG_OVERRIDE")
        is_exit_tags = key_u.endswith("_EXIT_TAGS")
        is_allowlist = key_u.endswith("_ALLOWLIST") or key_u.endswith("_UNIT_ALLOWLIST")

        if not (is_mode or is_tags or is_strategy_tag or is_exit_tags or is_allowlist):
            continue
        for token in values:
            if _looks_like_tag_value(token):
                tags.add(token)
    return tags


def _extract_modules_from_execstart(exec_line: str) -> list[str]:
    modules: list[str] = []
    try:
        parts = shlex.split(exec_line)
    except Exception:
        parts = exec_line.split()
    for i, part in enumerate(parts):
        if part == "-m" and i + 1 < len(parts):
            mod = parts[i + 1]
            if mod.startswith("workers."):
                modules.append(mod)
    for part in parts:
        if part.startswith("workers.") and part.endswith((".worker", ".exit_worker")):
            modules.append(part)
    return modules


def _module_roles(module: str) -> tuple[bool, bool]:
    if not module:
        return False, False
    lowered = module.lower()
    return lowered.endswith(".worker"), lowered.endswith(".exit_worker")


def _discover_from_systemd(systemd_dir: Path, running_services: set[str], now: dt.datetime) -> dict[str, StrategyRecord]:
    discovered: dict[str, StrategyRecord] = {}
    running = _systemctl_available()

    ignore = {
        "quant-strategy-control.service",
        "quant-market-data-feed.service",
        "quant-order-manager.service",
        "quant-position-manager.service",
        "quant-pattern-book.service",
        "quant-range-metrics.service",
        "quant-dynamic-alloc.service",
        "quant-ops-policy.service",
        "quant-policy-cycle.service",
        "quant-policy-guard.service",
        "quant-v2-audit.service",
        "quant-strategy-optimizer.service",
        "quant-entry-thesis-daily.service",
        "quant-autotune.service",
        "quant-boot-sync.service",
        "quant-maintain-logs.service",
        "quant-ui-snapshot.service",
        "quant-excursion-report.service",
        "quant-health-snapshot.service",
        "quant-type-maintenance.service",
        "quant-ssh-watchdog.service",
        "quant-level-map.service",
        "quant-bq-insights.service",
        "quant-bq-sync.service",
    }

    for name, body in _read_service_units(systemd_dir):
        if name in ignore:
            continue
        is_running = True
        if running:
            is_running = name in running_services

        env_paths: list[Path] = []
        modules: list[str] = []
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("Environment"):
                if line.startswith("ExecStart="):
                    modules.extend(_extract_modules_from_execstart(line.split("=", 1)[1]))
                continue

            if line.startswith("EnvironmentFile="):
                raw = line.split("=", 1)[1].strip()
                raw = raw.strip().strip('"').strip("'")
                if raw.startswith("-"):
                    raw = raw[1:]
                if raw:
                    env_paths.append(Path(raw))

        worker_tags: set[str] = set()
        if not env_paths:
            continue

        for path in env_paths:
            if path.name in {"quant-v2-runtime.env", "worker_autocontrol_off.env"}:
                continue
            env = _parse_env_file(path)
            worker_tags.update(_extract_tag_candidates_from_env(env))

            # mode-driven fallback (ex: tick_imbalance -> TickImbalance)
            for key, value in env.items():
                if not key.upper().endswith("_MODE"):
                    continue
                if _looks_like_tag_value(value):
                    worker_tags.add(value)

        fallback = False
        saw_entry_module = False
        saw_exit_module = False
        for module in modules:
            if not module.startswith("workers."):
                continue
            base = module.split(".")[-1]
            if not worker_tags and base == "session_open":
                worker_tags.add("session_open")
                fallback = True
            if not worker_tags and base == "scalp_precision":
                worker_tags.add("scalp_precision")
                fallback = True
            is_entry_module, is_exit_module = _module_roles(module)
            saw_entry_module = saw_entry_module or is_entry_module
            saw_exit_module = saw_exit_module or is_exit_module

        if not worker_tags:
            continue

        for tag in worker_tags:
            ckey = _norm_tag(tag)
            if not ckey:
                continue
            rec = discovered.setdefault(
                ckey,
                StrategyRecord(canonical_tag=ckey),
            )
            rec.sources.add(f"systemd:{name}")
            if saw_entry_module:
                rec.entry_active = rec.entry_active or is_running
            if saw_exit_module:
                rec.exit_active = rec.exit_active or is_running
            rec.active = rec.active or rec.entry_active or rec.exit_active
            if fallback:
                rec.sources.add("systemd_fallback")
            rec.last_closed = None

    return discovered


def _discover_from_control() -> dict[str, StrategyRecord]:
    discovered: dict[str, StrategyRecord] = {}
    try:
        from workers.common import strategy_control

        for slug, entry_enabled, *_ in strategy_control.list_enabled_strategies():
            ckey = _norm_tag(slug)
            if not ckey:
                continue
            rec = discovered.setdefault(ckey, StrategyRecord(canonical_tag=ckey))
            rec.sources.add("strategy_control")
            rec.enabled = bool(entry_enabled)
    except Exception:
        return {}
    return discovered


def _discover_from_trades(
    trades_db: Path,
    lookback_days: int,
) -> tuple[dict[str, StrategyStats], dict[str, str]]:
    if not trades_db.exists():
        return {}, {}
    conn = sqlite3.connect(f"file:{trades_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    since = f"-{max(1, int(lookback_days))} day"
    rows = conn.execute(
        """
        SELECT
          COALESCE(NULLIF(strategy_tag, ''), COALESCE(NULLIF(strategy, ''), 'unknown')) AS strategy,
          COUNT(*) AS trades,
          SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN pl_pips < 0 THEN 1 ELSE 0 END) AS losses,
          SUM(pl_pips) AS sum_pips,
          AVG(pl_pips) AS avg_pips,
          AVG(ABS(pl_pips)) AS avg_abs_pips,
          SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS gross_win,
          SUM(CASE WHEN pl_pips < 0 THEN -pl_pips ELSE 0 END) AS gross_loss,
          AVG(CASE WHEN open_time IS NOT NULL AND close_time IS NOT NULL
                  THEN (julianday(close_time) - julianday(open_time)) * 86400.0
                  ELSE NULL END) AS avg_hold_sec,
          MAX(close_time) AS last_closed
        FROM trades
        WHERE close_time IS NOT NULL
          AND strategy_tag IS NOT NULL
          AND close_time >= datetime('now', ?)
        GROUP BY 1
        ORDER BY strategy
        """,
        (since,),
    ).fetchall()
    stats_by_tag: dict[str, StrategyStats] = {}
    latest_by_tag: dict[str, str] = {}
    for row in rows:
        tag_raw = str(row["strategy"] or "")
        if not tag_raw:
            continue
        ckey = _norm_tag(tag_raw)
        if not ckey:
            continue
        latest = row["last_closed"]
        latest_by_tag[ckey] = str(latest or "")
        stats_by_tag[ckey] = StrategyStats(
            tag=tag_raw,
            trades=_to_int(row["trades"], 0),
            wins=_to_int(row["wins"], 0),
            losses=_to_int(row["losses"], 0),
            sum_pips=_to_float(row["sum_pips"], 0.0),
            avg_pips=_to_float(row["avg_pips"], 0.0),
            avg_abs_pips=_to_float(row["avg_abs_pips"], 0.0),
            gross_win=_to_float(row["gross_win"], 0.0),
            gross_loss=_to_float(row["gross_loss"], 0.0),
            avg_hold_sec=_to_optional_float(row["avg_hold_sec"]),
            last_closed=str(latest) if latest is not None else None,
        )
    conn.close()
    return stats_by_tag, latest_by_tag


def _select_squad(tag: str) -> str:
    normalized = tag.lower()
    if any(k in normalized for k in ("scalp", "ping", "m1scalper", "tick")):
        return "scalp"
    if "micro" in normalized:
        return "micro"
    if "macro" in normalized or "h1" in normalized:
        return "macro"
    if "session" in normalized:
        return "session"
    return "baseline"


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _squad_recommendation(tag: str, stats: StrategyStats, min_trades: int) -> dict[str, Any]:
    if stats.trades < min_trades:
        return {}

    confidence = _clamp(stats.trades / max(1, min_trades), 0.0, 1.0)
    wr = _clamp(stats.win_rate, 0.0, 1.0)
    pf = stats.profit_factor
    pf_score = 1.0 if pf == float("inf") else _clamp(pf, 0.0, 3.0)

    prob_multiplier = 1.0 + (wr - 0.5) * 0.35 * confidence
    units_multiplier = 1.0 + (pf_score - 1.0) * 0.22 * confidence
    tp_multiplier = 1.0 + (wr - 0.5) * 0.18 * confidence
    sl_multiplier = 1.0 - (wr - 0.5) * 0.12 * confidence

    if stats.loss_asymmetry == float("inf"):
        sl_multiplier *= 1.02
        prob_multiplier *= 0.96
    elif stats.loss_asymmetry > 1.5:
        sl_multiplier *= 0.95
        prob_multiplier *= 0.98
    elif stats.loss_asymmetry < 0.75:
        tp_multiplier *= 1.05

    avg_hold = stats.avg_hold_sec
    if avg_hold is not None:
        if avg_hold < 90 and stats.wins >= max(3, int(min_trades / 2)):
            units_multiplier *= 1.06
        if avg_hold > 600 and wr < 0.48:
            tp_multiplier *= 0.96

    squad = _select_squad(tag)
    if squad == "scalp":
        if wr >= 0.55 and stats.avg_abs_pips > 0.8:
            units_multiplier *= 1.04
            prob_multiplier *= 1.02
        if avg_hold is not None and avg_hold > 300:
            sl_multiplier *= 0.97
    elif squad == "micro":
        if wr <= 0.45:
            units_multiplier *= 0.92
            prob_multiplier *= 0.96
        if stats.avg_abs_pips < 1.0:
            tp_multiplier *= 0.95
    elif squad == "macro":
        if wr < 0.5:
            prob_multiplier *= 0.96
            units_multiplier *= 0.95
        if avg_hold is not None and avg_hold > 900:
            tp_multiplier *= 1.04
    elif squad == "session":
        units_multiplier *= 0.95
        sl_multiplier *= 0.96

    prob_multiplier = _clamp(prob_multiplier, 0.55, 1.28)
    units_multiplier = _clamp(units_multiplier, 0.55, 1.65)
    sl_multiplier = _clamp(sl_multiplier, 0.75, 1.25)
    tp_multiplier = _clamp(tp_multiplier, 0.75, 1.25)

    # emit only meaningful deltas to avoid noise
    out: dict[str, Any] = {
        "strategy_params": {
            "analysis_squad": squad,
            "win_rate": round(wr, 3),
            "profit_factor": None if pf == float("inf") else round(pf, 3),
            "trades": stats.trades,
            "avg_hold_sec": None if avg_hold is None else round(avg_hold, 1),
        },
    }
    if abs(prob_multiplier - 1.0) >= 0.02:
        out["entry_probability_multiplier"] = round(prob_multiplier, 4)
    if abs(units_multiplier - 1.0) >= 0.03:
        out["entry_units_multiplier"] = round(units_multiplier, 4)
    if abs(sl_multiplier - 1.0) >= 0.03:
        out["sl_distance_multiplier"] = round(sl_multiplier, 4)
    if abs(tp_multiplier - 1.0) >= 0.03:
        out["tp_distance_multiplier"] = round(tp_multiplier, 4)

    if len(out) <= 1:
        return {}
    return out


def _build_payload(config: WorkerConfig) -> dict[str, Any]:
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    running_services = _systemctl_running_services()
    discovered_systemd = _discover_from_systemd(config.systemd_dir, running_services, dt.datetime.utcnow())
    discovered_control = _discover_from_control()

    # merge discovery signals
    merged: dict[str, StrategyRecord] = {}
    for src in (discovered_control, discovered_systemd):
        for key, rec in src.items():
            merged.setdefault(key, StrategyRecord(canonical_tag=key))
            merged[key].sources.update(rec.sources)
            merged[key].entry_active = merged[key].entry_active or rec.entry_active
            merged[key].exit_active = merged[key].exit_active or rec.exit_active
            merged[key].active = merged[key].active or rec.active
            if rec.enabled is not None:
                merged[key].enabled = rec.enabled

    stats_by_tag, latest_by_tag = _discover_from_trades(config.trades_db, config.lookback_days)
    for key, stats in stats_by_tag.items():
        merged.setdefault(
            key,
            StrategyRecord(canonical_tag=key),
        )
        if stats.last_closed:
            merged[key].last_closed = str(stats.last_closed)
        merged[key].sources.add("trades")

    # apply latest close snapshot to missing canonical entries
    for key, close_time in latest_by_tag.items():
        if key in merged and not merged[key].last_closed:
            merged[key].last_closed = close_time

    payload: dict[str, Any] = {
        "version": "2026-02-24",
        "updated_at": now,
        "generated_by": "analysis/strategy_feedback_worker.py",
        "strategies": {},
    }

    if _systemctl_available():
        stale_limit = dt.timedelta(days=max(1, config.keep_inactive_days))
        cutoff = dt.datetime.utcnow() - stale_limit
    else:
        cutoff = None

    for tag, rec in sorted(merged.items(), key=lambda item: item[0]):
        stats = stats_by_tag.get(tag)
        if stats is None:
            continue

        if rec.enabled is False:
            continue

        if not rec.entry_active:
            # strategy workers removed/stopped: keep silent to avoid applying stale knobs
            if cutoff is not None and rec.last_closed:
                try:
                    if dt.datetime.fromisoformat(str(rec.last_closed).replace("Z", "+00:00")) < cutoff:
                        continue
                except Exception:
                    pass
            # no active worker and recently stopped: skip until it starts again
            continue

        advice = _squad_recommendation(tag, stats, config.min_trades)
        if not advice:
            continue

        # keep strategy-level metadata minimal and deterministic
        payload["strategies"][tag] = advice

    return payload


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strategy_feedback JSON")
    parser.add_argument("--nowrite", action="store_true", help="Do not write output file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    config = WorkerConfig()
    payload = _build_payload(config)

    if args.nowrite:
        logging.info("[strategy-feedback-worker] nowrite mode: %s", json.dumps(payload, ensure_ascii=False))
        return
    _write_payload(config.feedback_path, payload)
    logging.info(
        "[strategy-feedback-worker] wrote %s (%d strategies)",
        config.feedback_path,
        len(payload.get("strategies", {})),
    )


if __name__ == "__main__":
    main()
