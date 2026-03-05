#!/usr/bin/env python3
"""Apply local Brain model selection from benchmark report to env profile.

This script reads `scripts/benchmark_brain_local_llm.py` output, chooses:
- preflight model (quality + latency bounded)
- async autotune model (best quality)

and updates `ops/env/profiles/brain-ollama.env` keys accordingly.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REPORT = Path("logs/brain_local_llm_benchmark_latest.json")
DEFAULT_ENV_PROFILE = Path("ops/env/profiles/brain-ollama.env")
DEFAULT_OUTPUT = Path("logs/brain_model_selection_latest.json")


@dataclass(frozen=True)
class RankingRow:
    rank: int
    name: str
    model: str
    score: float
    parse_pass_rate: float
    alignment_coverage: float
    latency_p95_ms: float
    outcome_score: float
    outcome_scored_trades: int


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Brain model selection from benchmark JSON")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--env-profile", type=Path, default=DEFAULT_ENV_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-parse-pass-rate", type=float, default=0.9)
    parser.add_argument("--min-alignment-coverage", type=float, default=0.7)
    parser.add_argument("--max-preflight-latency-ms", type=float, default=9000.0)
    parser.add_argument("--fallback-preflight-model", default="qwen2.5:7b")
    parser.add_argument("--fallback-autotune-model", default="gpt-oss:20b")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_ranking(path: Path) -> tuple[dict[str, Any], list[RankingRow]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ranking_raw = payload.get("ranking")
    if not isinstance(ranking_raw, list) or not ranking_raw:
        raise ValueError(f"No ranking found in benchmark report: {path}")

    rows: list[RankingRow] = []
    for idx, item in enumerate(ranking_raw, start=1):
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or "").strip()
        if not model:
            continue
        rows.append(
            RankingRow(
                rank=_safe_int(item.get("rank"), idx),
                name=str(item.get("name") or model).strip(),
                model=model,
                score=_safe_float(item.get("score"), 0.0),
                parse_pass_rate=_safe_float(item.get("parse_pass_rate"), 0.0),
                alignment_coverage=_safe_float(item.get("alignment_coverage"), 0.0),
                latency_p95_ms=_safe_float(item.get("latency_p95_ms"), 0.0),
                outcome_score=_safe_float(item.get("outcome_score"), 0.0),
                outcome_scored_trades=_safe_int(item.get("outcome_scored_trades"), 0),
            )
        )
    if not rows:
        raise ValueError(f"No valid ranking rows in benchmark report: {path}")
    return payload, rows


def _pick_models(
    rows: list[RankingRow],
    *,
    min_parse_pass_rate: float,
    min_alignment_coverage: float,
    max_preflight_latency_ms: float,
    fallback_preflight_model: str,
    fallback_autotune_model: str,
) -> tuple[str, str, float]:
    quality_rows = [
        row
        for row in rows
        if row.parse_pass_rate >= min_parse_pass_rate and row.alignment_coverage >= min_alignment_coverage
    ]
    if not quality_rows:
        quality_rows = [row for row in rows if row.parse_pass_rate >= min_parse_pass_rate]
    if not quality_rows:
        quality_rows = list(rows)

    preflight_candidates = [
        row for row in quality_rows if row.latency_p95_ms > 0.0 and row.latency_p95_ms <= max_preflight_latency_ms
    ]
    if preflight_candidates:
        preflight = max(preflight_candidates, key=lambda row: (row.score, -row.latency_p95_ms))
    else:
        with_latency = [row for row in quality_rows if row.latency_p95_ms > 0.0]
        preflight = min(with_latency, key=lambda row: row.latency_p95_ms) if with_latency else quality_rows[0]

    autotune = max(quality_rows, key=lambda row: (row.score, row.parse_pass_rate, -row.latency_p95_ms))

    preflight_model = preflight.model or str(fallback_preflight_model).strip()
    autotune_model = autotune.model or str(fallback_autotune_model).strip()
    if not preflight_model:
        preflight_model = str(fallback_preflight_model).strip() or "qwen2.5:7b"
    if not autotune_model:
        autotune_model = str(fallback_autotune_model).strip() or preflight_model

    return preflight_model, autotune_model, max(preflight.latency_p95_ms, 0.0)


def _recommend_timeout_sec(latency_p95_ms: float) -> int:
    if latency_p95_ms <= 0.0:
        return 8
    recommended = int(math.ceil((latency_p95_ms / 1000.0) * 2.0))
    return max(4, min(20, recommended))


def _apply_env_updates(existing_text: str, updates: dict[str, str]) -> str:
    lines = existing_text.splitlines()
    remaining = dict(updates)

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, _value = line.split("=", 1)
        key = key.strip()
        if key not in remaining:
            continue
        lines[idx] = f"{key}={remaining.pop(key)}"

    if remaining:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([f"{key}={value}" for key, value in remaining.items()])

    return "\n".join(lines).rstrip() + "\n"


def _parse_env_values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = raw_line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = _parse_args()
    benchmark, rows = _load_ranking(args.benchmark)

    preflight_model, autotune_model, preflight_latency = _pick_models(
        rows,
        min_parse_pass_rate=float(args.min_parse_pass_rate),
        min_alignment_coverage=float(args.min_alignment_coverage),
        max_preflight_latency_ms=float(args.max_preflight_latency_ms),
        fallback_preflight_model=str(args.fallback_preflight_model),
        fallback_autotune_model=str(args.fallback_autotune_model),
    )
    timeout_sec = _recommend_timeout_sec(preflight_latency)

    updates = {
        "BRAIN_OLLAMA_MODEL": preflight_model,
        "BRAIN_PROMPT_AUTO_TUNE_MODEL": autotune_model,
        "BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL": autotune_model,
        "BRAIN_TIMEOUT_SEC": str(timeout_sec),
    }

    env_before = args.env_profile.read_text(encoding="utf-8") if args.env_profile.exists() else ""
    env_after = _apply_env_updates(env_before, updates)
    env_before_values = _parse_env_values(env_before)
    changed_keys = sorted([key for key, value in updates.items() if env_before_values.get(key) != value])
    env_changed = bool(changed_keys)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark_path": str(args.benchmark),
        "env_profile_path": str(args.env_profile),
        "preflight_model": preflight_model,
        "autotune_model": autotune_model,
        "preflight_timeout_sec": timeout_sec,
        "selection_constraints": {
            "min_parse_pass_rate": float(args.min_parse_pass_rate),
            "min_alignment_coverage": float(args.min_alignment_coverage),
            "max_preflight_latency_ms": float(args.max_preflight_latency_ms),
        },
        "ranking": [asdict(row) for row in rows],
        "benchmark_generated_at": benchmark.get("generated_at"),
        "selected_source": benchmark.get("selected_source"),
        "updated_keys": updates,
        "changed_keys": changed_keys,
        "env_changed": env_changed,
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        if env_after != env_before:
            _write_atomic(args.env_profile, env_after)
        _write_atomic(args.output, json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))

    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
