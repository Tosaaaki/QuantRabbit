#!/usr/bin/env python3
"""Scheduled worker wrapper for replay quality gate runs.

This worker executes `scripts/replay_quality_gate.py`, stores the latest status
snapshot, appends run history, trims old run artifacts, and can connect replay
output to analysis feedback for automatic tuning updates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable

from utils.market_hours import is_market_open

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency during local tests
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
_AUTO_IMPROVE_SCOPE_VALUES = {"failing", "all"}
_AUTO_IMPROVE_WORKER_SKIP_PREFIXES = ("pocket:", "source:")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def _resolve_path(text: str, *, base_dir: Path) -> Path:
    path = Path(str(text).strip())
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _split_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _normalize_hours(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for value in raw:
        hour = _safe_int(value, default=-1)
        if 0 <= hour <= 23 and hour not in out:
            out.append(hour)
    out.sort()
    return out


def _safe_filename(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def _tail(text: str | None, *, limit: int = 12000) -> str:
    raw = str(text or "")
    return raw[-limit:]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _iter_report_paths(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    rows: list[Path] = []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        report = child / "quality_gate_report.json"
        if report.exists():
            rows.append(report)
    rows.sort(key=lambda p: (p.stat().st_mtime, p.as_posix()), reverse=True)
    return rows


def _extract_report_path_from_stdout(stdout: str, *, out_dir: Path) -> Path | None:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate or not candidate.endswith("quality_gate_report.json"):
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if path.exists():
            return path
    reports = _iter_report_paths(out_dir)
    if reports:
        return reports[0]
    return None


def _cleanup_old_runs(out_dir: Path, *, keep_runs: int, protected: Path | None = None) -> list[str]:
    if keep_runs <= 0 or not out_dir.exists():
        return []
    reports = _iter_report_paths(out_dir)
    protected_dir = protected.parent.resolve() if protected is not None else None

    keep_dirs: set[Path] = set()
    for report in reports[:keep_runs]:
        keep_dirs.add(report.parent.resolve())
    if protected_dir is not None:
        keep_dirs.add(protected_dir)

    removed: list[str] = []
    for report in reports[keep_runs:]:
        run_dir = report.parent.resolve()
        if run_dir in keep_dirs:
            continue
        shutil.rmtree(run_dir, ignore_errors=True)
        removed.append(run_dir.name)
    return removed


@dataclass(frozen=True)
class WorkerConfig:
    config_path: Path
    out_dir: Path
    state_path: Path
    history_path: Path
    timeout_sec: int
    keep_runs: int
    strict: bool
    ticks_glob: str
    workers: str
    backend: str
    auto_improve_enabled: bool = False
    auto_improve_scope: str = "failing"
    auto_improve_strategies: str = ""
    auto_improve_include_live_trades: bool = False
    auto_improve_replay_json_globs: str = ""
    auto_improve_counterfactual_timeout_sec: int = 600
    auto_improve_counterfactual_out_dir: Path = (REPO_ROOT / "logs" / "replay_auto_improve").resolve()
    auto_improve_min_trades: int = 40
    auto_improve_max_block_hours: int = 8
    auto_improve_apply_block_hours: bool = False
    auto_improve_min_reentry_confidence: float = 0.70
    auto_improve_min_reentry_lcb_uplift_pips: float = 0.20
    auto_improve_min_apply_interval_sec: int = 10800
    auto_improve_apply_state_path: Path = (REPO_ROOT / "logs" / "replay_auto_improve_state.json").resolve()
    auto_improve_reentry_config_path: Path = (REPO_ROOT / "config" / "worker_reentry.yaml").resolve()
    auto_improve_apply_reentry: bool = True


def _default_config() -> WorkerConfig:
    scope = str(os.getenv("REPLAY_QUALITY_GATE_AUTO_IMPROVE_SCOPE", "failing")).strip().lower()
    if scope not in _AUTO_IMPROVE_SCOPE_VALUES:
        scope = "failing"
    return WorkerConfig(
        config_path=_resolve_path(
            os.getenv("REPLAY_QUALITY_GATE_CONFIG", "config/replay_quality_gate.yaml"),
            base_dir=REPO_ROOT,
        ),
        out_dir=_resolve_path(
            os.getenv("REPLAY_QUALITY_GATE_OUT_DIR", "tmp/replay_quality_gate"),
            base_dir=REPO_ROOT,
        ),
        state_path=_resolve_path(
            os.getenv("REPLAY_QUALITY_GATE_STATE_PATH", "logs/replay_quality_gate_latest.json"),
            base_dir=REPO_ROOT,
        ),
        history_path=_resolve_path(
            os.getenv("REPLAY_QUALITY_GATE_HISTORY_PATH", "logs/replay_quality_gate_history.jsonl"),
            base_dir=REPO_ROOT,
        ),
        timeout_sec=max(30, _env_int("REPLAY_QUALITY_GATE_TIMEOUT_SEC", 1800)),
        keep_runs=max(1, _env_int("REPLAY_QUALITY_GATE_KEEP_RUNS", 24)),
        strict=_env_bool("REPLAY_QUALITY_GATE_STRICT", False),
        ticks_glob=str(os.getenv("REPLAY_QUALITY_GATE_TICKS_GLOB", "")).strip(),
        workers=str(os.getenv("REPLAY_QUALITY_GATE_WORKERS", "")).strip(),
        backend=str(os.getenv("REPLAY_QUALITY_GATE_BACKEND", "")).strip(),
        auto_improve_enabled=_env_bool("REPLAY_QUALITY_GATE_AUTO_IMPROVE_ENABLED", False),
        auto_improve_scope=scope,
        auto_improve_strategies=str(os.getenv("REPLAY_QUALITY_GATE_AUTO_IMPROVE_STRATEGIES", "")).strip(),
        auto_improve_include_live_trades=_env_bool(
            "REPLAY_QUALITY_GATE_AUTO_IMPROVE_INCLUDE_LIVE_TRADES",
            False,
        ),
        auto_improve_replay_json_globs=str(
            os.getenv("REPLAY_QUALITY_GATE_AUTO_IMPROVE_REPLAY_JSON_GLOBS", "")
        ).strip(),
        auto_improve_counterfactual_timeout_sec=max(
            60,
            _env_int("REPLAY_QUALITY_GATE_AUTO_IMPROVE_COUNTERFACTUAL_TIMEOUT_SEC", 600),
        ),
        auto_improve_counterfactual_out_dir=_resolve_path(
            os.getenv("REPLAY_QUALITY_GATE_AUTO_IMPROVE_COUNTERFACTUAL_OUT_DIR", "logs/replay_auto_improve"),
            base_dir=REPO_ROOT,
        ),
        auto_improve_min_trades=max(
            1,
            _env_int("REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_TRADES", 40),
        ),
        auto_improve_max_block_hours=max(
            1,
            _env_int("REPLAY_QUALITY_GATE_AUTO_IMPROVE_MAX_BLOCK_HOURS", 8),
        ),
        auto_improve_apply_block_hours=_env_bool(
            "REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_BLOCK_HOURS",
            False,
        ),
        auto_improve_min_reentry_confidence=_clamp(
            _safe_float(os.getenv("REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_REENTRY_CONFIDENCE"), 0.70),
            0.0,
            1.0,
        ),
        auto_improve_min_reentry_lcb_uplift_pips=_safe_float(
            os.getenv("REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_REENTRY_LCB_UPLIFT_PIPS"),
            0.20,
        ),
        auto_improve_min_apply_interval_sec=max(
            0,
            _env_int("REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_APPLY_INTERVAL_SEC", 10800),
        ),
        auto_improve_apply_state_path=_resolve_path(
            os.getenv(
                "REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_STATE_PATH",
                "logs/replay_auto_improve_state.json",
            ),
            base_dir=REPO_ROOT,
        ),
        auto_improve_reentry_config_path=_resolve_path(
            os.getenv("REPLAY_QUALITY_GATE_AUTO_IMPROVE_REENTRY_CONFIG_PATH", "config/worker_reentry.yaml"),
            base_dir=REPO_ROOT,
        ),
        auto_improve_apply_reentry=_env_bool("REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_REENTRY", True),
    )


def parse_args() -> argparse.Namespace:
    default = _default_config()
    ap = argparse.ArgumentParser(description="Replay quality gate scheduled worker")
    ap.add_argument("--config", type=Path, default=default.config_path)
    ap.add_argument("--out-dir", type=Path, default=default.out_dir)
    ap.add_argument("--state-path", type=Path, default=default.state_path)
    ap.add_argument("--history-path", type=Path, default=default.history_path)
    ap.add_argument("--timeout-sec", type=int, default=default.timeout_sec)
    ap.add_argument("--keep-runs", type=int, default=default.keep_runs)
    ap.add_argument("--strict", action="store_true", default=default.strict)
    ap.add_argument("--ticks-glob", default=default.ticks_glob)
    ap.add_argument("--workers", default=default.workers)
    ap.add_argument(
        "--backend",
        choices=("exit_workers_groups", "exit_workers_main", ""),
        default=default.backend,
    )
    ap.add_argument(
        "--auto-improve-enabled",
        type=int,
        choices=(0, 1),
        default=1 if default.auto_improve_enabled else 0,
    )
    ap.add_argument(
        "--auto-improve-scope",
        choices=("failing", "all"),
        default=default.auto_improve_scope,
    )
    ap.add_argument("--auto-improve-strategies", default=default.auto_improve_strategies)
    ap.add_argument(
        "--auto-improve-include-live-trades",
        type=int,
        choices=(0, 1),
        default=1 if default.auto_improve_include_live_trades else 0,
    )
    ap.add_argument("--auto-improve-replay-json-globs", default=default.auto_improve_replay_json_globs)
    ap.add_argument(
        "--auto-improve-counterfactual-timeout-sec",
        type=int,
        default=default.auto_improve_counterfactual_timeout_sec,
    )
    ap.add_argument(
        "--auto-improve-counterfactual-out-dir",
        type=Path,
        default=default.auto_improve_counterfactual_out_dir,
    )
    ap.add_argument("--auto-improve-min-trades", type=int, default=default.auto_improve_min_trades)
    ap.add_argument(
        "--auto-improve-reentry-config-path",
        type=Path,
        default=default.auto_improve_reentry_config_path,
    )
    ap.add_argument("--auto-improve-max-block-hours", type=int, default=default.auto_improve_max_block_hours)
    ap.add_argument(
        "--auto-improve-apply-block-hours",
        type=int,
        choices=(0, 1),
        default=1 if default.auto_improve_apply_block_hours else 0,
    )
    ap.add_argument(
        "--auto-improve-min-reentry-confidence",
        type=float,
        default=default.auto_improve_min_reentry_confidence,
    )
    ap.add_argument(
        "--auto-improve-min-reentry-lcb-uplift-pips",
        type=float,
        default=default.auto_improve_min_reentry_lcb_uplift_pips,
    )
    ap.add_argument(
        "--auto-improve-min-apply-interval-sec",
        type=int,
        default=default.auto_improve_min_apply_interval_sec,
    )
    ap.add_argument(
        "--auto-improve-apply-state-path",
        type=Path,
        default=default.auto_improve_apply_state_path,
    )
    ap.add_argument(
        "--auto-improve-apply-reentry",
        type=int,
        choices=(0, 1),
        default=1 if default.auto_improve_apply_reentry else 0,
    )
    return ap.parse_args()


def _build_command(cfg: WorkerConfig) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "replay_quality_gate.py"),
        "--config",
        str(cfg.config_path),
        "--out-dir",
        str(cfg.out_dir),
        "--timeout-sec",
        str(max(30, int(cfg.timeout_sec))),
    ]
    if cfg.strict:
        cmd.append("--strict")
    if cfg.ticks_glob:
        cmd.extend(["--ticks-glob", cfg.ticks_glob])
    if cfg.workers:
        cmd.extend(["--workers", cfg.workers])
    if cfg.backend:
        cmd.extend(["--backend", cfg.backend])
    return cmd


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_sec: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=max(30, int(timeout_sec)),
        check=False,
    )


def _summarize_report(report_payload: dict[str, Any] | None) -> tuple[str, list[str], int]:
    if not isinstance(report_payload, dict):
        return "unknown", [], 0
    overall = report_payload.get("overall") if isinstance(report_payload.get("overall"), dict) else {}
    meta = report_payload.get("meta") if isinstance(report_payload.get("meta"), dict) else {}
    status = str(overall.get("status") or "unknown")
    failing_workers = overall.get("failing_workers")
    if not isinstance(failing_workers, list):
        failing_workers = []
    failing = [str(v) for v in failing_workers if str(v).strip()]
    folds = 0
    try:
        folds = int(meta.get("fold_count") or 0)
    except Exception:
        folds = 0
    return status, failing, folds


def _build_config_from_args(args: argparse.Namespace) -> WorkerConfig:
    scope = str(args.auto_improve_scope or "").strip().lower()
    if scope not in _AUTO_IMPROVE_SCOPE_VALUES:
        scope = "failing"
    return WorkerConfig(
        config_path=Path(args.config).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        state_path=Path(args.state_path).resolve(),
        history_path=Path(args.history_path).resolve(),
        timeout_sec=max(30, int(args.timeout_sec)),
        keep_runs=max(1, int(args.keep_runs)),
        strict=bool(args.strict),
        ticks_glob=str(args.ticks_glob or "").strip(),
        workers=str(args.workers or "").strip(),
        backend=str(args.backend or "").strip(),
        auto_improve_enabled=bool(int(args.auto_improve_enabled)),
        auto_improve_scope=scope,
        auto_improve_strategies=str(args.auto_improve_strategies or "").strip(),
        auto_improve_include_live_trades=bool(int(args.auto_improve_include_live_trades)),
        auto_improve_replay_json_globs=str(args.auto_improve_replay_json_globs or "").strip(),
        auto_improve_counterfactual_timeout_sec=max(60, int(args.auto_improve_counterfactual_timeout_sec)),
        auto_improve_counterfactual_out_dir=Path(args.auto_improve_counterfactual_out_dir).resolve(),
        auto_improve_min_trades=max(1, int(args.auto_improve_min_trades)),
        auto_improve_max_block_hours=max(1, int(args.auto_improve_max_block_hours)),
        auto_improve_apply_block_hours=bool(int(args.auto_improve_apply_block_hours)),
        auto_improve_min_reentry_confidence=_clamp(
            float(args.auto_improve_min_reentry_confidence),
            0.0,
            1.0,
        ),
        auto_improve_min_reentry_lcb_uplift_pips=float(args.auto_improve_min_reentry_lcb_uplift_pips),
        auto_improve_min_apply_interval_sec=max(0, int(args.auto_improve_min_apply_interval_sec)),
        auto_improve_apply_state_path=Path(args.auto_improve_apply_state_path).resolve(),
        auto_improve_reentry_config_path=Path(args.auto_improve_reentry_config_path).resolve(),
        auto_improve_apply_reentry=bool(int(args.auto_improve_apply_reentry)),
    )


def _collect_auto_improve_strategies(
    report_payload: dict[str, Any] | None,
    cfg: WorkerConfig,
) -> list[str]:
    explicit = _dedupe_keep_order(_split_csv(cfg.auto_improve_strategies))
    if explicit:
        return explicit
    if not isinstance(report_payload, dict):
        return []
    meta = report_payload.get("meta") if isinstance(report_payload.get("meta"), dict) else {}
    overall = report_payload.get("overall") if isinstance(report_payload.get("overall"), dict) else {}
    workers = meta.get("workers") if isinstance(meta.get("workers"), list) else []
    failing_workers = overall.get("failing_workers") if isinstance(overall.get("failing_workers"), list) else []
    selected = failing_workers if cfg.auto_improve_scope == "failing" else workers
    candidates = [str(worker).strip() for worker in selected if str(worker).strip()]
    out: list[str] = []
    for candidate in candidates:
        lower = candidate.lower()
        if lower == "__overall__":
            continue
        if lower.startswith(_AUTO_IMPROVE_WORKER_SKIP_PREFIXES):
            continue
        out.append(candidate)
    return _dedupe_keep_order(out)


def _resolve_auto_improve_replay_globs(
    cfg: WorkerConfig,
    report_path: Path | None,
) -> list[str]:
    override = _dedupe_keep_order(_split_csv(cfg.auto_improve_replay_json_globs))
    if override:
        return [str(_resolve_path(pattern, base_dir=REPO_ROOT)) for pattern in override]
    if report_path is None:
        return []
    run_root = report_path.parent / "runs"
    return [
        str(run_root / "*" / "replay_exit_workers.json"),
        str(run_root / "*" / "replay_exit_*_base.json"),
    ]


def _build_counterfactual_command(
    *,
    strategy_like: str,
    replay_json_globs: list[str],
    out_path: Path,
    history_path: Path,
    include_live_trades: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "analysis.trade_counterfactual_worker",
        "--strategy-like",
        strategy_like,
        "--include-live-trades",
        "1" if include_live_trades else "0",
        "--replay-json-globs",
        ",".join(replay_json_globs),
        "--out-path",
        str(out_path),
        "--history-path",
        str(history_path),
    ]
    return cmd


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    if yaml is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("PyYAML is not available")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


def _parse_iso8601(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    text = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_apply_state(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def _write_apply_state(
    *,
    path: Path,
    applied_at: datetime,
    updates: dict[str, Any],
) -> None:
    payload = {
        "applied_at": applied_at.isoformat(),
        "updates": updates,
    }
    _write_json_atomic(path, payload)


def _normalize_bias(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"favor", "avoid", "neutral"}:
        return text
    return "neutral"


def _safe_positive_int(value: Any, default: int) -> int:
    parsed = _safe_int(value, default=default)
    if parsed <= 0:
        return int(default)
    return int(parsed)


def _apply_reentry_updates(
    *,
    reentry_path: Path,
    strategy_updates: dict[str, dict[str, Any]],
    apply_block_hours: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(reentry_path),
        "requested_updates": strategy_updates,
        "applied": False,
        "changed": [],
        "reason": "",
    }
    if yaml is None:
        result["reason"] = "yaml_unavailable"
        return result
    if not strategy_updates:
        result["reason"] = "no_strategy_updates"
        return result

    base = _load_yaml_dict(reentry_path)
    defaults = base.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {}
    strategies = base.get("strategies")
    if not isinstance(strategies, dict):
        strategies = {}
        base["strategies"] = strategies

    changed: list[dict[str, Any]] = []
    for strategy, update in strategy_updates.items():
        if not isinstance(update, dict):
            continue
        row = strategies.get(strategy)
        if not isinstance(row, dict):
            row = {}
            strategies[strategy] = row
        row_changes: list[dict[str, Any]] = []

        if apply_block_hours:
            hours = _normalize_hours(update.get("block_jst_hours"))
            if hours:
                current_hours = _normalize_hours(row.get("block_jst_hours"))
                if current_hours != hours:
                    row["block_jst_hours"] = hours
                    row_changes.append(
                        {
                            "key": "block_jst_hours",
                            "before": current_hours,
                            "after": hours,
                        }
                    )

        hint = update.get("reentry_overrides")
        if not isinstance(hint, dict):
            if row_changes:
                changed.append({"strategy": strategy, "changes": row_changes})
            continue

        mode = str(hint.get("mode") or "").strip().lower()
        if mode not in {"tighten", "loosen"}:
            if row_changes:
                changed.append({"strategy": strategy, "changes": row_changes})
            continue

        current_cooldown_loss = _safe_positive_int(
            row.get("cooldown_loss_sec", defaults.get("cooldown_loss_sec", 180)),
            180,
        )
        current_cooldown_win = _safe_positive_int(
            row.get("cooldown_win_sec", defaults.get("cooldown_win_sec", 60)),
            60,
        )
        current_reentry_pips = abs(
            _safe_float(
                row.get("same_dir_reentry_pips", defaults.get("same_dir_reentry_pips", 1.8)),
                1.8,
            )
        )
        if current_reentry_pips <= 0.0:
            current_reentry_pips = 1.8

        cooldown_loss_mult = _clamp(_safe_float(hint.get("cooldown_loss_mult"), 1.0), 0.60, 1.80)
        cooldown_win_mult = _clamp(_safe_float(hint.get("cooldown_win_mult"), 1.0), 0.70, 1.50)
        reentry_pips_mult = _clamp(
            _safe_float(hint.get("same_dir_reentry_pips_mult"), 1.0),
            0.70,
            1.60,
        )
        target_bias = _normalize_bias(
            hint.get("return_wait_bias") or ("avoid" if mode == "tighten" else "favor")
        )

        target_cooldown_loss = int(round(current_cooldown_loss * cooldown_loss_mult))
        target_cooldown_win = int(round(current_cooldown_win * cooldown_win_mult))
        target_reentry_pips = current_reentry_pips * reentry_pips_mult
        target_cooldown_loss = max(10, min(3600, target_cooldown_loss))
        target_cooldown_win = max(5, min(1800, target_cooldown_win))
        target_reentry_pips = _clamp(target_reentry_pips, 0.05, 25.0)

        if target_cooldown_loss != current_cooldown_loss:
            row["cooldown_loss_sec"] = target_cooldown_loss
            row_changes.append(
                {
                    "key": "cooldown_loss_sec",
                    "before": current_cooldown_loss,
                    "after": target_cooldown_loss,
                }
            )
        if target_cooldown_win != current_cooldown_win:
            row["cooldown_win_sec"] = target_cooldown_win
            row_changes.append(
                {
                    "key": "cooldown_win_sec",
                    "before": current_cooldown_win,
                    "after": target_cooldown_win,
                }
            )
        if abs(target_reentry_pips - current_reentry_pips) >= 1e-4:
            target_reentry_rounded = round(target_reentry_pips, 3)
            row["same_dir_reentry_pips"] = target_reentry_rounded
            row_changes.append(
                {
                    "key": "same_dir_reentry_pips",
                    "before": round(current_reentry_pips, 3),
                    "after": target_reentry_rounded,
                }
            )
        current_bias = _normalize_bias(row.get("return_wait_bias", defaults.get("return_wait_bias", "neutral")))
        if target_bias != current_bias:
            row["return_wait_bias"] = target_bias
            row_changes.append(
                {
                    "key": "return_wait_bias",
                    "before": current_bias,
                    "after": target_bias,
                }
            )

        if row_changes:
            changed.append({"strategy": strategy, "changes": row_changes})

    if not changed:
        result["reason"] = "no_diff"
        return result

    _write_yaml_atomic(reentry_path, base)
    result["applied"] = True
    result["changed"] = changed
    result["reason"] = "applied"
    return result


def _run_auto_improve(
    cfg: WorkerConfig,
    *,
    replay_returncode: int,
    report_payload: dict[str, Any] | None,
    report_path: Path | None,
    counterfactual_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": bool(cfg.auto_improve_enabled),
        "status": "disabled",
        "reason": "",
        "scope": cfg.auto_improve_scope,
        "replay_json_globs": [],
        "strategies": [],
        "strategy_runs": [],
        "accepted_updates": {},
        "reentry_apply": {},
    }
    if not cfg.auto_improve_enabled:
        result["reason"] = "feature_disabled"
        return result
    if replay_returncode != 0:
        result["status"] = "skipped"
        result["reason"] = "replay_failed"
        return result
    if report_path is None:
        result["status"] = "skipped"
        result["reason"] = "report_missing"
        return result

    strategies = _collect_auto_improve_strategies(report_payload, cfg)
    result["strategies"] = list(strategies)
    if not strategies:
        result["status"] = "skipped"
        result["reason"] = "no_target_strategies"
        return result

    replay_json_globs = _resolve_auto_improve_replay_globs(cfg, report_path)
    result["replay_json_globs"] = list(replay_json_globs)
    if not replay_json_globs:
        result["status"] = "skipped"
        result["reason"] = "replay_json_globs_empty"
        return result

    out_root = (cfg.auto_improve_counterfactual_out_dir / report_path.parent.name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    executor = counterfactual_runner or (
        lambda c: _run_subprocess(
            c,
            cwd=REPO_ROOT,
            timeout_sec=cfg.auto_improve_counterfactual_timeout_sec,
        )
    )
    accepted_updates: dict[str, dict[str, Any]] = {}
    strategy_runs: list[dict[str, Any]] = []

    for strategy in strategies:
        stem = _safe_filename(strategy)
        out_path = out_root / f"{stem}.json"
        history_path = out_root / f"{stem}.history.jsonl"
        cmd = _build_counterfactual_command(
            strategy_like=strategy,
            replay_json_globs=replay_json_globs,
            out_path=out_path,
            history_path=history_path,
            include_live_trades=cfg.auto_improve_include_live_trades,
        )
        proc = executor(cmd)
        row: dict[str, Any] = {
            "strategy": strategy,
            "status": "",
            "returncode": int(proc.returncode),
            "counterfactual_out_path": str(out_path),
            "counterfactual_history_path": str(history_path),
            "command": cmd,
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
        }
        if proc.returncode != 0:
            row["status"] = "counterfactual_error"
            strategy_runs.append(row)
            continue

        payload = _read_json(out_path)
        if not isinstance(payload, dict):
            row["status"] = "counterfactual_output_missing"
            strategy_runs.append(row)
            continue

        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        hints = payload.get("policy_hints") if isinstance(payload.get("policy_hints"), dict) else {}
        trades = _safe_int(summary.get("trades"), default=0)
        block_hours = _normalize_hours(hints.get("block_jst_hours"))
        stuck_ratio = _safe_float(summary.get("stuck_trade_ratio"), default=0.0)
        reentry_hint = hints.get("reentry_overrides") if isinstance(hints.get("reentry_overrides"), dict) else {}
        reentry_mode = str(reentry_hint.get("mode") or "").strip().lower()
        reentry_confidence = _clamp(_safe_float(reentry_hint.get("confidence"), 0.0), 0.0, 1.0)
        reentry_lcb = max(0.0, _safe_float(reentry_hint.get("lcb_uplift_pips"), 0.0))

        row["trades"] = trades
        row["stuck_trade_ratio"] = stuck_ratio
        row["block_jst_hours"] = block_hours
        row["reentry_mode"] = reentry_mode or "unknown"
        row["reentry_confidence"] = round(reentry_confidence, 4)
        row["reentry_lcb_uplift_pips"] = round(reentry_lcb, 4)

        if trades < cfg.auto_improve_min_trades:
            row["status"] = "skipped_min_trades"
            row["min_trades"] = int(cfg.auto_improve_min_trades)
            strategy_runs.append(row)
            continue

        strategy_update: dict[str, Any] = {}
        if reentry_mode in {"tighten", "loosen"}:
            if reentry_confidence < cfg.auto_improve_min_reentry_confidence:
                row["status"] = "skipped_low_reentry_confidence"
                row["min_reentry_confidence"] = round(cfg.auto_improve_min_reentry_confidence, 4)
            elif reentry_lcb < cfg.auto_improve_min_reentry_lcb_uplift_pips:
                row["status"] = "skipped_low_reentry_lcb"
                row["min_reentry_lcb_uplift_pips"] = round(cfg.auto_improve_min_reentry_lcb_uplift_pips, 4)
            else:
                strategy_update["reentry_overrides"] = {
                    "mode": reentry_mode,
                    "confidence": reentry_confidence,
                    "lcb_uplift_pips": reentry_lcb,
                    "cooldown_loss_mult": _clamp(
                        _safe_float(reentry_hint.get("cooldown_loss_mult"), 1.0), 0.60, 1.80
                    ),
                    "cooldown_win_mult": _clamp(
                        _safe_float(reentry_hint.get("cooldown_win_mult"), 1.0), 0.70, 1.50
                    ),
                    "same_dir_reentry_pips_mult": _clamp(
                        _safe_float(reentry_hint.get("same_dir_reentry_pips_mult"), 1.0),
                        0.70,
                        1.60,
                    ),
                    "return_wait_bias": _normalize_bias(reentry_hint.get("return_wait_bias")),
                    "source": str(reentry_hint.get("source") or "counterfactual"),
                }

        if cfg.auto_improve_apply_block_hours:
            if block_hours:
                if len(block_hours) <= int(cfg.auto_improve_max_block_hours):
                    strategy_update["block_jst_hours"] = block_hours
                else:
                    row["block_hours_status"] = "ignored_too_many_block_hours"
                    row["max_block_hours"] = int(cfg.auto_improve_max_block_hours)
            else:
                row["block_hours_status"] = "ignored_no_block_hours"

        if not strategy_update:
            if not row.get("status"):
                row["status"] = "skipped_no_reentry_overrides"
            strategy_runs.append(row)
            continue

        accepted_updates[strategy] = strategy_update
        row["accepted_update"] = strategy_update
        row["status"] = "accepted"
        strategy_runs.append(row)

    result["strategy_runs"] = strategy_runs
    result["accepted_updates"] = accepted_updates

    if not accepted_updates:
        result["status"] = "analyzed"
        result["reason"] = "no_accepted_updates"
        return result

    if not cfg.auto_improve_apply_reentry:
        result["status"] = "analyzed"
        result["reason"] = "reentry_apply_disabled"
        return result

    if int(cfg.auto_improve_min_apply_interval_sec) > 0:
        now_utc = datetime.now(timezone.utc)
        apply_state = _read_apply_state(cfg.auto_improve_apply_state_path)
        last_applied_at = _parse_iso8601(apply_state.get("applied_at"))
        if last_applied_at is not None:
            elapsed_sec = max(0, int((now_utc - last_applied_at).total_seconds()))
            if elapsed_sec < int(cfg.auto_improve_min_apply_interval_sec):
                result["status"] = "analyzed"
                result["reason"] = "reentry_apply_cooldown"
                result["reentry_apply"] = {
                    "applied": False,
                    "path": str(cfg.auto_improve_reentry_config_path),
                    "state_path": str(cfg.auto_improve_apply_state_path),
                    "last_applied_at": last_applied_at.isoformat(),
                    "elapsed_sec": elapsed_sec,
                    "min_apply_interval_sec": int(cfg.auto_improve_min_apply_interval_sec),
                    "remaining_sec": int(cfg.auto_improve_min_apply_interval_sec) - elapsed_sec,
                }
                return result

    apply_result = _apply_reentry_updates(
        reentry_path=cfg.auto_improve_reentry_config_path,
        strategy_updates=accepted_updates,
        apply_block_hours=cfg.auto_improve_apply_block_hours,
    )
    result["reentry_apply"] = apply_result
    if bool(apply_result.get("applied")):
        _write_apply_state(
            path=cfg.auto_improve_apply_state_path,
            applied_at=datetime.now(timezone.utc),
            updates=accepted_updates,
        )
        result["status"] = "applied"
        result["reason"] = "reentry_updated"
    else:
        result["status"] = "analyzed"
        result["reason"] = str(apply_result.get("reason") or "reentry_not_updated")
    return result


def run_once(
    cfg: WorkerConfig,
    *,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    counterfactual_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> int:
    started = datetime.now(timezone.utc)
    cmd = _build_command(cfg)
    executor = runner or (lambda c: _run_subprocess(c, cwd=REPO_ROOT, timeout_sec=cfg.timeout_sec))
    proc = executor(cmd)
    finished = datetime.now(timezone.utc)

    report_path = _extract_report_path_from_stdout(proc.stdout or "", out_dir=cfg.out_dir)
    report_payload = _read_json(report_path) if report_path is not None else None
    gate_status, failing_workers, fold_count = _summarize_report(report_payload)
    md_path = report_path.with_name("quality_gate_report.md") if report_path else None
    auto_improve = _run_auto_improve(
        cfg,
        replay_returncode=int(proc.returncode),
        report_payload=report_payload,
        report_path=report_path,
        counterfactual_runner=counterfactual_runner,
    )
    removed_dirs = _cleanup_old_runs(cfg.out_dir, keep_runs=cfg.keep_runs, protected=report_path)

    payload = {
        "worker": "replay_quality_gate_worker",
        "generated_at": finished.isoformat(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_sec": (finished - started).total_seconds(),
        "returncode": int(proc.returncode),
        "strict": bool(cfg.strict),
        "gate_status": gate_status,
        "failing_workers": failing_workers,
        "fold_count": int(fold_count),
        "config_path": str(cfg.config_path),
        "out_dir": str(cfg.out_dir),
        "report_json_path": str(report_path) if report_path else "",
        "report_md_path": str(md_path) if md_path and md_path.exists() else "",
        "cleaned_run_dirs": removed_dirs,
        "command": cmd,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
        "auto_improve": auto_improve,
    }
    _write_json_atomic(cfg.state_path, payload)
    _append_jsonl(cfg.history_path, payload)

    print(
        f"[replay-quality-gate-worker] rc={proc.returncode} gate_status={gate_status} "
        f"report={payload['report_json_path'] or '-'} auto_improve={auto_improve.get('status', 'disabled')}"
    )
    return int(proc.returncode)


def main() -> int:
    if not _env_bool("REPLAY_QUALITY_GATE_ENABLED", True):
        print("[replay-quality-gate-worker] skipped: REPLAY_QUALITY_GATE_ENABLED=0")
        return 0
    if _env_bool("REPLAY_QUALITY_GATE_SKIP_WHEN_MARKET_OPEN", False) and is_market_open():
        print("[replay-quality-gate-worker] skipped: market_open")
        return 0
    args = parse_args()
    cfg = _build_config_from_args(args)
    return run_once(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
