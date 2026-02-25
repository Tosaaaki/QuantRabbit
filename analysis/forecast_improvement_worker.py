#!/usr/bin/env python3
"""Scheduled worker for forecast improvement audits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]

_EVAL_OVERRIDE_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "--feature-expansion-gain",
        (
            "FORECAST_IMPROVEMENT_AUDIT_FEATURE_EXPANSION_GAIN",
            "FORECAST_TECH_FEATURE_EXPANSION_GAIN",
        ),
    ),
    (
        "--breakout-adaptive-weight",
        (
            "FORECAST_IMPROVEMENT_AUDIT_BREAKOUT_ADAPTIVE_WEIGHT",
            "FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT",
        ),
    ),
    (
        "--breakout-adaptive-weight-map",
        (
            "FORECAST_IMPROVEMENT_AUDIT_BREAKOUT_ADAPTIVE_WEIGHT_MAP",
            "FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP",
        ),
    ),
    (
        "--breakout-adaptive-min-samples",
        (
            "FORECAST_IMPROVEMENT_AUDIT_BREAKOUT_ADAPTIVE_MIN_SAMPLES",
            "FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES",
        ),
    ),
    (
        "--breakout-adaptive-lookback",
        (
            "FORECAST_IMPROVEMENT_AUDIT_BREAKOUT_ADAPTIVE_LOOKBACK",
            "FORECAST_TECH_BREAKOUT_ADAPTIVE_LOOKBACK",
        ),
    ),
    (
        "--session-bias-weight",
        (
            "FORECAST_IMPROVEMENT_AUDIT_SESSION_BIAS_WEIGHT",
            "FORECAST_TECH_SESSION_BIAS_WEIGHT",
        ),
    ),
    (
        "--session-bias-weight-map",
        (
            "FORECAST_IMPROVEMENT_AUDIT_SESSION_BIAS_WEIGHT_MAP",
            "FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP",
        ),
    ),
    (
        "--session-bias-min-samples",
        (
            "FORECAST_IMPROVEMENT_AUDIT_SESSION_BIAS_MIN_SAMPLES",
            "FORECAST_TECH_SESSION_BIAS_MIN_SAMPLES",
        ),
    ),
    (
        "--session-bias-lookback",
        (
            "FORECAST_IMPROVEMENT_AUDIT_SESSION_BIAS_LOOKBACK",
            "FORECAST_TECH_SESSION_BIAS_LOOKBACK",
        ),
    ),
    (
        "--rebound-weight",
        (
            "FORECAST_IMPROVEMENT_AUDIT_REBOUND_WEIGHT",
            "FORECAST_TECH_REBOUND_WEIGHT",
        ),
    ),
    (
        "--rebound-weight-map",
        (
            "FORECAST_IMPROVEMENT_AUDIT_REBOUND_WEIGHT_MAP",
            "FORECAST_TECH_REBOUND_WEIGHT_MAP",
        ),
    ),
    (
        "--dynamic-weight-enabled",
        (
            "FORECAST_IMPROVEMENT_AUDIT_DYNAMIC_WEIGHT_ENABLED",
            "FORECAST_TECH_DYNAMIC_WEIGHT_ENABLED",
        ),
    ),
    (
        "--dynamic-weight-horizons",
        (
            "FORECAST_IMPROVEMENT_AUDIT_DYNAMIC_WEIGHT_HORIZONS",
            "FORECAST_TECH_DYNAMIC_WEIGHT_HORIZONS",
        ),
    ),
    (
        "--dynamic-max-scale-delta",
        (
            "FORECAST_IMPROVEMENT_AUDIT_DYNAMIC_MAX_SCALE_DELTA",
            "FORECAST_TECH_DYNAMIC_MAX_SCALE_DELTA",
        ),
    ),
    (
        "--dynamic-breakout-skill-center",
        (
            "FORECAST_IMPROVEMENT_AUDIT_DYNAMIC_BREAKOUT_SKILL_CENTER",
            "FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_CENTER",
        ),
    ),
    (
        "--dynamic-breakout-skill-gain",
        (
            "FORECAST_IMPROVEMENT_AUDIT_DYNAMIC_BREAKOUT_SKILL_GAIN",
            "FORECAST_TECH_DYNAMIC_BREAKOUT_SKILL_GAIN",
        ),
    ),
    (
        "--dynamic-breakout-regime-gain",
        (
            "FORECAST_IMPROVEMENT_AUDIT_DYNAMIC_BREAKOUT_REGIME_GAIN",
            "FORECAST_TECH_DYNAMIC_BREAKOUT_REGIME_GAIN",
        ),
    ),
    (
        "--dynamic-session-bias-center",
        (
            "FORECAST_IMPROVEMENT_AUDIT_DYNAMIC_SESSION_BIAS_CENTER",
            "FORECAST_TECH_DYNAMIC_SESSION_BIAS_CENTER",
        ),
    ),
    (
        "--dynamic-session-bias-gain",
        (
            "FORECAST_IMPROVEMENT_AUDIT_DYNAMIC_SESSION_BIAS_GAIN",
            "FORECAST_TECH_DYNAMIC_SESSION_BIAS_GAIN",
        ),
    ),
    (
        "--dynamic-session-regime-gain",
        (
            "FORECAST_IMPROVEMENT_AUDIT_DYNAMIC_SESSION_REGIME_GAIN",
            "FORECAST_TECH_DYNAMIC_SESSION_REGIME_GAIN",
        ),
    ),
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _resolve_path(text: str) -> Path:
    path = Path(str(text).strip())
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _split_csv(raw: str) -> tuple[str, ...]:
    values = []
    for token in str(raw).split(","):
        item = token.strip()
        if item:
            values.append(item)
    return tuple(values)


def _tail(text: str | None, *, limit: int = 12000) -> str:
    raw = str(text or "")
    return raw[-limit:]


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
        temp_path = Path(fh.name)
    temp_path.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
        temp_path = Path(fh.name)
    temp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _find_first_env(keys: tuple[str, ...]) -> str:
    for key in keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        value = str(raw).strip()
        if value:
            return value
    return ""


def _normalize_zero_one(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return "1"
    if text in {"0", "false", "no", "off"}:
        return "0"
    return str(value).strip()


def _collect_eval_overrides() -> list[str]:
    args: list[str] = []
    for flag, keys in _EVAL_OVERRIDE_KEYS:
        value = _find_first_env(keys)
        if not value:
            continue
        if flag == "--dynamic-weight-enabled":
            value = _normalize_zero_one(value)
        args.extend([flag, value])
    return args


@dataclass(frozen=True)
class WorkerConfig:
    env_file: Path
    out_dir: Path
    state_path: Path
    history_path: Path
    latest_md_path: Path
    patterns: str
    steps: str
    horizons: tuple[str, ...]
    max_bars: int
    timeout_sec: int
    keep_runs: int
    hit_degrade_threshold: float
    mae_degrade_threshold: float
    range_cov_degrade_threshold: float


def _default_config() -> WorkerConfig:
    out_dir = _resolve_path(
        os.getenv(
            "FORECAST_IMPROVEMENT_AUDIT_OUT_DIR",
            "logs/reports/forecast_improvement",
        )
    )
    return WorkerConfig(
        env_file=_resolve_path(
            os.getenv(
                "FORECAST_IMPROVEMENT_AUDIT_ENV_FILE",
                "ops/env/quant-v2-runtime.env",
            )
        ),
        out_dir=out_dir,
        state_path=_resolve_path(
            os.getenv(
                "FORECAST_IMPROVEMENT_AUDIT_STATE_PATH",
                "logs/forecast_improvement_latest.json",
            )
        ),
        history_path=_resolve_path(
            os.getenv(
                "FORECAST_IMPROVEMENT_AUDIT_HISTORY_PATH",
                "logs/forecast_improvement_history.jsonl",
            )
        ),
        latest_md_path=_resolve_path(
            os.getenv(
                "FORECAST_IMPROVEMENT_AUDIT_LATEST_MD_PATH",
                str(out_dir / "latest.md"),
            )
        ),
        patterns=str(
            os.getenv(
                "FORECAST_IMPROVEMENT_AUDIT_PATTERNS",
                "logs/candles_M1*.json,logs/candles_USDJPY_M1*.json,logs/oanda/candles_M1_latest.json",
            )
        ).strip(),
        steps=str(os.getenv("FORECAST_IMPROVEMENT_AUDIT_STEPS", "1,5,10")).strip(),
        horizons=_split_csv(os.getenv("FORECAST_IMPROVEMENT_AUDIT_HORIZONS", "1m,5m,10m")),
        max_bars=max(0, _env_int("FORECAST_IMPROVEMENT_AUDIT_MAX_BARS", 8050)),
        timeout_sec=max(60, _env_int("FORECAST_IMPROVEMENT_AUDIT_TIMEOUT_SEC", 900)),
        keep_runs=max(1, _env_int("FORECAST_IMPROVEMENT_AUDIT_KEEP_RUNS", 96)),
        hit_degrade_threshold=_env_float("FORECAST_IMPROVEMENT_AUDIT_HIT_DEGRADE_THRESHOLD", -0.002),
        mae_degrade_threshold=_env_float("FORECAST_IMPROVEMENT_AUDIT_MAE_DEGRADE_THRESHOLD", 0.020),
        range_cov_degrade_threshold=_env_float(
            "FORECAST_IMPROVEMENT_AUDIT_RANGE_COV_DEGRADE_THRESHOLD",
            -0.030,
        ),
    )


def parse_args() -> argparse.Namespace:
    default = _default_config()
    ap = argparse.ArgumentParser(description="Forecast improvement scheduled worker")
    ap.add_argument("--env-file", type=Path, default=default.env_file)
    ap.add_argument("--out-dir", type=Path, default=default.out_dir)
    ap.add_argument("--state-path", type=Path, default=default.state_path)
    ap.add_argument("--history-path", type=Path, default=default.history_path)
    ap.add_argument("--latest-md-path", type=Path, default=default.latest_md_path)
    ap.add_argument("--patterns", default=default.patterns)
    ap.add_argument("--steps", default=default.steps)
    ap.add_argument("--horizons", default=",".join(default.horizons))
    ap.add_argument("--max-bars", type=int, default=default.max_bars)
    ap.add_argument("--timeout-sec", type=int, default=default.timeout_sec)
    ap.add_argument("--keep-runs", type=int, default=default.keep_runs)
    ap.add_argument("--hit-degrade-threshold", type=float, default=default.hit_degrade_threshold)
    ap.add_argument("--mae-degrade-threshold", type=float, default=default.mae_degrade_threshold)
    ap.add_argument(
        "--range-cov-degrade-threshold",
        type=float,
        default=default.range_cov_degrade_threshold,
    )
    return ap.parse_args()


def _build_config_from_args(args: argparse.Namespace) -> WorkerConfig:
    horizons = _split_csv(str(args.horizons or "1m,5m,10m"))
    return WorkerConfig(
        env_file=Path(args.env_file).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        state_path=Path(args.state_path).resolve(),
        history_path=Path(args.history_path).resolve(),
        latest_md_path=Path(args.latest_md_path).resolve(),
        patterns=str(args.patterns or "").strip(),
        steps=str(args.steps or "").strip(),
        horizons=horizons if horizons else ("1m", "5m", "10m"),
        max_bars=max(0, int(args.max_bars)),
        timeout_sec=max(60, int(args.timeout_sec)),
        keep_runs=max(1, int(args.keep_runs)),
        hit_degrade_threshold=float(args.hit_degrade_threshold),
        mae_degrade_threshold=float(args.mae_degrade_threshold),
        range_cov_degrade_threshold=float(args.range_cov_degrade_threshold),
    )


def _build_snapshot_command(cfg: WorkerConfig) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "vm_forecast_snapshot.py"),
        "--env-file",
        str(cfg.env_file),
        "--json",
    ]
    for horizon in cfg.horizons:
        cmd.extend(["--horizon", str(horizon)])
    return cmd


def _build_eval_command(cfg: WorkerConfig, *, eval_json_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_forecast_before_after.py"),
        "--patterns",
        cfg.patterns,
        "--steps",
        cfg.steps,
    ]
    if cfg.max_bars > 0:
        cmd.extend(["--max-bars", str(cfg.max_bars)])
    cmd.extend(_collect_eval_overrides())
    cmd.extend(["--json-out", str(eval_json_path)])
    return cmd


def _run_subprocess(cmd: list[str], *, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=max(60, int(timeout_sec)),
        check=False,
    )


def _parse_snapshot_rows(text: str) -> dict[str, dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    payload: Any
    try:
        payload = json.loads(raw)
    except Exception:
        first = raw.find("[")
        last = raw.rfind("]")
        if first < 0 or last < 0 or last <= first:
            return {}
        try:
            payload = json.loads(raw[first : last + 1])
        except Exception:
            return {}
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict):
                horizon = str(item[0]).strip().lower()
                if horizon:
                    rows[horizon] = dict(item[1])
                continue
            if isinstance(item, dict):
                horizon = str(item.get("horizon") or "").strip().lower()
                if horizon:
                    rows[horizon] = dict(item)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                rows[str(key).strip().lower()] = dict(value)
    return rows


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_eval_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append(dict(row))
    return normalized


def _classify_verdict(
    rows: list[dict[str, Any]],
    *,
    hit_threshold: float,
    mae_threshold: float,
    range_cov_threshold: float,
) -> tuple[str, list[str], list[str]]:
    if not rows:
        return "unknown", [], []
    improved: list[str] = []
    degraded: list[str] = []
    for row in rows:
        horizon = str(row.get("horizon") or row.get("step") or "").strip().lower()
        hit_delta = _to_float(row.get("hit_delta"))
        mae_delta = _to_float(row.get("mae_delta"))
        range_cov_delta = _to_float(row.get("range_coverage_delta"))
        if (
            hit_delta < hit_threshold
            or mae_delta > mae_threshold
            or range_cov_delta < range_cov_threshold
        ):
            degraded.append(horizon or "unknown")
            continue
        if hit_delta >= 0.0 and mae_delta <= 0.0 and range_cov_delta >= 0.0:
            improved.append(horizon or "unknown")

    if degraded:
        return "degraded", sorted(set(improved)), sorted(set(degraded))
    if len(improved) == len(rows):
        return "improved", sorted(set(improved)), []
    return "mixed", sorted(set(improved)), []


def _build_next_actions(verdict: str, degraded_horizons: list[str]) -> list[str]:
    if verdict == "degraded":
        targets = ",".join(degraded_horizons) if degraded_horizons else "degraded horizons"
        return [
            f"Reduce `FORECAST_TECH_FEATURE_EXPANSION_GAIN` by 0.05 for {targets} and re-evaluate.",
            f"Lower `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP` weights for {targets} by 0.04, then rerun.",
            "Increase `FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES` by +20 to reduce short-window overreaction.",
        ]
    if verdict == "improved":
        return [
            "Keep current runtime forecast weights and rerun with a wider `FORECAST_IMPROVEMENT_AUDIT_MAX_BARS` window for stability confirmation.",
        ]
    return [
        "Adjust only mixed horizons in `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP` with 0.02-0.04 local changes.",
        "Keep non-degraded horizons fixed and rerun to isolate the delta source.",
        "If range coverage is flat, tune `FORECAST_RANGE_BAND_LOWER_Q/UPPER_Q` and compare again on the same bars.",
    ]


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except Exception:
        return "-"
    return f"{f:.{digits}f}"


def _sort_horizon_key(horizon: str) -> tuple[int, int, str]:
    text = str(horizon or "").strip().lower()
    if not text:
        return (99, 0, "")
    try:
        value = int(text[:-1])
        unit = text[-1]
    except Exception:
        return (99, 0, text)
    order = {"m": 0, "h": 1, "d": 2, "w": 3}.get(unit, 4)
    return (order, value, text)


def _build_report_markdown(
    *,
    generated_at: str,
    cfg: WorkerConfig,
    run_dir: Path,
    snapshot_rows: dict[str, dict[str, Any]],
    eval_payload: dict[str, Any] | None,
    verdict: str,
    improved_horizons: list[str],
    degraded_horizons: list[str],
    actions: list[str],
    eval_json_path: Path,
) -> str:
    rows = _extract_eval_rows(eval_payload)
    summary = eval_payload.get("summary") if isinstance(eval_payload, dict) else {}
    bars = summary.get("bars") if isinstance(summary, dict) else None
    from_ts = summary.get("from_ts") if isinstance(summary, dict) else None
    to_ts = summary.get("to_ts") if isinstance(summary, dict) else None
    lines: list[str] = [
        "# Forecast Improvement Audit",
        "",
        "## Execution",
        f"- generated_at: `{generated_at}`",
        f"- horizons: `{','.join(cfg.horizons)}`",
        f"- steps: `{cfg.steps}`",
        f"- patterns: `{cfg.patterns}`",
        f"- max_bars: `{cfg.max_bars}`",
        f"- bars: `{bars if bars is not None else '-'}`",
        f"- period: `{from_ts or '-'} -> {to_ts or '-'}`",
        f"- run_dir: `{run_dir}`",
        f"- eval_json: `{eval_json_path}`",
        "",
        "## Snapshot",
    ]
    if snapshot_rows:
        for horizon in sorted(snapshot_rows.keys(), key=_sort_horizon_key):
            row = snapshot_rows[horizon]
            lines.append(
                "- "
                f"{horizon}: p_up={_fmt(row.get('p_up'))} "
                f"edge={_fmt(row.get('edge'))} "
                f"expected_pips={_fmt(row.get('expected_pips'))} "
                f"as_of={row.get('feature_ts') or '-'}"
            )
    else:
        lines.append("- snapshot unavailable")

    lines.extend(["", "## Before/After"])
    if rows:
        for row in sorted(rows, key=lambda r: _sort_horizon_key(str(r.get("horizon") or ""))):
            horizon = str(row.get("horizon") or row.get("step") or "-").lower()
            lines.append(
                "- "
                f"{horizon}: "
                f"hit_before={_fmt(row.get('hit_before'))} "
                f"hit_after={_fmt(row.get('hit_after'))} "
                f"hit_delta={_fmt(row.get('hit_delta'), 4)} "
                f"mae_before={_fmt(row.get('mae_before'))} "
                f"mae_after={_fmt(row.get('mae_after'))} "
                f"mae_delta={_fmt(row.get('mae_delta'), 4)} "
                f"range_cov_before={_fmt(row.get('range_coverage_before'))} "
                f"range_cov_after={_fmt(row.get('range_coverage_after'))} "
                f"range_cov_delta={_fmt(row.get('range_coverage_delta'), 4)}"
            )
    else:
        lines.append("- before/after rows unavailable")

    lines.extend(
        [
            "",
            "## Verdict",
            f"- verdict: **{verdict}**",
            f"- improved_horizons: `{','.join(improved_horizons) if improved_horizons else '-'}`",
            f"- degraded_horizons: `{','.join(degraded_horizons) if degraded_horizons else '-'}`",
            "",
            "## Next Actions",
        ]
    )
    for action in actions[:3]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def _iter_report_paths(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    reports: list[Path] = []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        report = child / "report.md"
        if report.exists():
            reports.append(report)
    reports.sort(key=lambda p: (p.stat().st_mtime, p.as_posix()), reverse=True)
    return reports


def _cleanup_old_runs(out_dir: Path, *, keep_runs: int, protected: Path | None = None) -> list[str]:
    if keep_runs <= 0:
        return []
    reports = _iter_report_paths(out_dir)
    if len(reports) <= keep_runs:
        return []
    keep_dirs: set[Path] = set()
    for report in reports[:keep_runs]:
        keep_dirs.add(report.parent.resolve())
    if protected is not None:
        keep_dirs.add(protected.parent.resolve())
    removed: list[str] = []
    for report in reports[keep_runs:]:
        target = report.parent.resolve()
        if target in keep_dirs:
            continue
        shutil.rmtree(target, ignore_errors=True)
        removed.append(target.name)
    return removed


Runner = Callable[[list[str], int], subprocess.CompletedProcess[str]]


def run_once(cfg: WorkerConfig, *, runner: Runner | None = None) -> int:
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    run_dir = cfg.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_json_path = run_dir / "snapshot.json"
    eval_json_path = run_dir / "forecast_eval.json"
    report_md_path = run_dir / "report.md"

    execute = runner or (lambda cmd, timeout: _run_subprocess(cmd, timeout_sec=timeout))

    snapshot_cmd = _build_snapshot_command(cfg)
    snapshot_proc = execute(snapshot_cmd, cfg.timeout_sec)
    snapshot_rows = _parse_snapshot_rows(snapshot_proc.stdout or "")
    if snapshot_rows:
        _write_json_atomic(snapshot_json_path, snapshot_rows)
    _write_text_atomic(run_dir / "snapshot.stdout.log", snapshot_proc.stdout or "")
    _write_text_atomic(run_dir / "snapshot.stderr.log", snapshot_proc.stderr or "")

    eval_cmd = _build_eval_command(cfg, eval_json_path=eval_json_path)
    eval_proc = execute(eval_cmd, cfg.timeout_sec)
    _write_text_atomic(run_dir / "eval.stdout.log", eval_proc.stdout or "")
    _write_text_atomic(run_dir / "eval.stderr.log", eval_proc.stderr or "")

    eval_payload = _read_json(eval_json_path)
    rows = _extract_eval_rows(eval_payload)
    verdict, improved_horizons, degraded_horizons = _classify_verdict(
        rows,
        hit_threshold=cfg.hit_degrade_threshold,
        mae_threshold=cfg.mae_degrade_threshold,
        range_cov_threshold=cfg.range_cov_degrade_threshold,
    )
    actions = _build_next_actions(verdict, degraded_horizons)
    report_md = _build_report_markdown(
        generated_at=started.isoformat(),
        cfg=cfg,
        run_dir=run_dir,
        snapshot_rows=snapshot_rows,
        eval_payload=eval_payload,
        verdict=verdict,
        improved_horizons=improved_horizons,
        degraded_horizons=degraded_horizons,
        actions=actions,
        eval_json_path=eval_json_path,
    )
    _write_text_atomic(report_md_path, report_md)
    _write_text_atomic(cfg.latest_md_path, report_md)

    removed_dirs = _cleanup_old_runs(cfg.out_dir, keep_runs=cfg.keep_runs, protected=report_md_path)
    finished = datetime.now(timezone.utc)

    has_rows = bool(rows)
    if eval_proc.returncode == 0 and has_rows:
        returncode = 0
    elif eval_proc.returncode != 0:
        returncode = int(eval_proc.returncode)
    else:
        returncode = 2

    payload = {
        "worker": "forecast_improvement_worker",
        "generated_at": finished.isoformat(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_sec": (finished - started).total_seconds(),
        "returncode": returncode,
        "verdict": verdict,
        "improved_horizons": improved_horizons,
        "degraded_horizons": degraded_horizons,
        "rows_count": len(rows),
        "snapshot_rows_count": len(snapshot_rows),
        "snapshot_returncode": int(snapshot_proc.returncode),
        "eval_returncode": int(eval_proc.returncode),
        "run_dir": str(run_dir),
        "snapshot_json_path": str(snapshot_json_path) if snapshot_json_path.exists() else "",
        "eval_json_path": str(eval_json_path) if eval_json_path.exists() else "",
        "report_md_path": str(report_md_path),
        "latest_md_path": str(cfg.latest_md_path),
        "patterns": cfg.patterns,
        "steps": cfg.steps,
        "horizons": list(cfg.horizons),
        "max_bars": int(cfg.max_bars),
        "thresholds": {
            "hit_degrade_threshold": cfg.hit_degrade_threshold,
            "mae_degrade_threshold": cfg.mae_degrade_threshold,
            "range_cov_degrade_threshold": cfg.range_cov_degrade_threshold,
        },
        "snapshot_command": snapshot_cmd,
        "eval_command": eval_cmd,
        "cleaned_run_dirs": removed_dirs,
        "snapshot_stdout_tail": _tail(snapshot_proc.stdout),
        "snapshot_stderr_tail": _tail(snapshot_proc.stderr),
        "eval_stdout_tail": _tail(eval_proc.stdout),
        "eval_stderr_tail": _tail(eval_proc.stderr),
    }
    _write_json_atomic(cfg.state_path, payload)
    _append_jsonl(cfg.history_path, payload)

    print(
        "[forecast-improvement-worker] "
        f"rc={returncode} verdict={verdict} rows={len(rows)} report={report_md_path}"
    )
    return returncode


def main() -> int:
    if not _env_bool("FORECAST_IMPROVEMENT_AUDIT_ENABLED", True):
        print("[forecast-improvement-worker] skipped: FORECAST_IMPROVEMENT_AUDIT_ENABLED=0")
        return 0
    args = parse_args()
    cfg = _build_config_from_args(args)
    return run_once(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
