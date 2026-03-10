#!/usr/bin/env python3
"""Compute soft participation boosts/trims from recent entry-path conversion."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.strategy_tags import extract_strategy_tags, resolve_strategy_tag


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def _load_recent_realized_jpy(trades_db: Path, *, lookback_hours: float) -> dict[str, float]:
    if not trades_db.exists():
        return {}
    out: dict[str, float] = {}
    con = sqlite3.connect(f"file:{trades_db}?mode=ro", uri=True, timeout=8.0, isolation_level=None)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT strategy_tag, strategy, entry_thesis, COALESCE(realized_pl, 0.0)
            FROM trades
            WHERE close_time IS NOT NULL
              AND julianday(close_time) >= julianday('now', ?)
            """,
            (f"-{max(0.1, float(lookback_hours)):.3f} hours",),
        )
        for strategy_tag, strategy, entry_thesis, realized_pl in cur.fetchall():
            strategy_key, _canonical_key = extract_strategy_tags(
                strategy_tag=strategy_tag,
                strategy=strategy,
                entry_thesis=entry_thesis,
            )
            if not strategy_key:
                strategy_key = resolve_strategy_tag(str(strategy_tag or strategy or "").strip()) or str(
                    strategy_tag or strategy or ""
                ).strip()
            if not strategy_key:
                continue
            out[strategy_key] = out.get(strategy_key, 0.0) + _safe_float(realized_pl, 0.0)
    finally:
        con.close()
    return out


def build_participation_alloc(
    entry_path_summary: dict[str, Any],
    *,
    realized_by_strategy: dict[str, float],
    min_attempts: int,
    max_units_cut: float,
    max_units_boost: float,
    max_prob_boost: float,
) -> dict[str, Any]:
    strategies = entry_path_summary.get("strategies")
    if not isinstance(strategies, dict):
        strategies = {}

    fill_rates = [
        _safe_float(rec.get("filled_rate"), 0.0)
        for rec in strategies.values()
        if isinstance(rec, dict) and int(rec.get("attempts") or 0) >= min_attempts
    ]
    median_fill_rate = _median(fill_rates)
    if median_fill_rate <= 0.0:
        median_fill_rate = 0.02

    output_strategies: dict[str, Any] = {}
    for raw_key, record in sorted(strategies.items()):
        if not isinstance(record, dict):
            continue
        strategy_key = resolve_strategy_tag(str(raw_key or "").strip()) or str(raw_key or "").strip()
        attempts = int(record.get("attempts") or 0)
        fills = int(record.get("fills") or 0)
        filled_rate = _safe_float(record.get("filled_rate"), 0.0)
        attempt_share = _safe_float(record.get("attempt_share"), 0.0)
        fill_share = _safe_float(record.get("fill_share"), 0.0)
        share_gap = _safe_float(record.get("share_gap"), attempt_share - fill_share)
        realized_jpy = _safe_float(realized_by_strategy.get(strategy_key), 0.0)
        units_multiplier = 1.0
        probability_multiplier = 1.0
        probability_boost = 0.0
        cadence_floor = 1.0
        action = "hold"
        terminal_status_counts = record.get("terminal_status_counts") if isinstance(record.get("terminal_status_counts"), dict) else {}
        hard_block_rate = 0.0
        if terminal_status_counts:
            hard_blocks = 0
            for status_name, count in terminal_status_counts.items():
                if str(status_name or "").strip() in {"perf_block", "entry_probability_reject", "rejected"}:
                    hard_blocks += int(count or 0)
            hard_block_rate = hard_blocks / max(1, attempts)

        if attempts >= max(1, min_attempts):
            if share_gap >= 0.08 and filled_rate <= (median_fill_rate * 0.85) and realized_jpy <= 0.0:
                severity = _clamp((share_gap - 0.08) / 0.22, 0.0, 1.0)
                severity = max(severity, _clamp((median_fill_rate - filled_rate) / max(0.01, median_fill_rate), 0.0, 1.0))
                units_multiplier = 1.0 - max_units_cut * (0.35 + 0.65 * severity)
                cadence_floor = 0.90
                action = "trim_units"
            elif fill_share >= attempt_share + 0.02 and filled_rate >= median_fill_rate and realized_jpy >= 0.0:
                advantage = _clamp((fill_share - attempt_share) / 0.18, 0.0, 1.0)
                quality = _clamp((filled_rate - median_fill_rate) / max(0.01, median_fill_rate), 0.0, 1.0)
                boost = max_units_boost * (0.30 + 0.70 * max(advantage, quality))
                units_multiplier = 1.0 + boost
                probability_boost = max_prob_boost * (0.25 + 0.75 * max(advantage, quality))
                cadence_floor = 1.00
                action = "boost_participation"
            elif filled_rate >= median_fill_rate * 1.10 and realized_jpy >= 0.0 and attempts >= max(min_attempts, 8):
                units_multiplier = 1.0 + max_units_boost * 0.40
                probability_boost = max_prob_boost * 0.35
                cadence_floor = 1.00
                action = "boost_participation"

        quality_score = _clamp(
            0.45 * _clamp(filled_rate / max(0.01, median_fill_rate), 0.0, 1.25)
            + 0.30 * _clamp(fill_share / max(0.01, attempt_share) if attempt_share > 0 else 0.0, 0.0, 1.25)
            + 0.25 * _clamp((realized_jpy + 500.0) / 1500.0, 0.0, 1.0),
            0.0,
            1.25,
        )

        output_strategies[str(strategy_key or raw_key)] = {
            "strategy_key": str(strategy_key or raw_key),
            "pocket": str(record.get("pocket") or "").strip() or "unknown",
            "attempts": attempts,
            "preflights": attempts,
            "fills": fills,
            "filled": fills,
            "filled_rate": round(filled_rate, 6),
            "fill_rate": round(filled_rate, 6),
            "attempt_share": round(attempt_share, 6),
            "current_share": round(attempt_share, 6),
            "fill_share": round(fill_share, 6),
            "target_share": round(fill_share, 6),
            "share_gap": round(share_gap, 6),
            "realized_jpy": round(realized_jpy, 3),
            "units_multiplier": round(_clamp(units_multiplier, 1.0 - max_units_cut, 1.0 + max_units_boost), 4),
            "lot_multiplier": round(_clamp(units_multiplier, 1.0 - max_units_cut, 1.0 + max_units_boost), 4),
            "probability_multiplier": round(max(1.0, probability_multiplier), 4),
            "probability_offset": round(_clamp(probability_boost, 0.0, max_prob_boost), 4),
            "probability_boost": round(_clamp(probability_boost, 0.0, max_prob_boost), 4),
            "cadence_floor": round(_clamp(cadence_floor, 0.85, 1.0), 4),
            "quality_score": round(quality_score, 4),
            "hard_block_rate": round(hard_block_rate, 6),
            "action": action,
        }

    action_counts: dict[str, int] = {}
    for rec in output_strategies.values():
        action = str(rec.get("action") or "hold")
        action_counts[action] = action_counts.get(action, 0) + 1

    return {
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "lookback_hours": _safe_float(entry_path_summary.get("lookback_hours"), 24.0),
        "median_filled_rate": round(median_fill_rate, 6),
        "allocation_policy": {
            "protect_frequency": True,
            "min_attempts": int(min_attempts),
            "max_units_cut": round(max_units_cut, 4),
            "max_units_boost": round(max_units_boost, 4),
            "max_probability_boost": round(max_prob_boost, 4),
            "negative_probability_offsets_enabled": False,
        },
        "action_counts": action_counts,
        "strategies": output_strategies,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build soft participation allocator artifact")
    ap.add_argument("--entry-path-summary", default="logs/entry_path_summary_latest.json")
    ap.add_argument("--trades-db", default="logs/trades.db")
    ap.add_argument("--output", default="config/participation_alloc.json")
    ap.add_argument("--lookback-hours", type=float, default=24.0)
    ap.add_argument("--min-attempts", type=int, default=20)
    ap.add_argument("--max-units-cut", type=float, default=0.18)
    ap.add_argument("--max-units-boost", type=float, default=0.12)
    ap.add_argument("--max-probability-boost", type=float, default=0.05)
    args = ap.parse_args()

    summary = _read_json(Path(args.entry_path_summary).resolve())
    realized_by_strategy = _load_recent_realized_jpy(
        Path(args.trades_db).resolve(),
        lookback_hours=float(args.lookback_hours),
    )
    payload = build_participation_alloc(
        summary,
        realized_by_strategy=realized_by_strategy,
        min_attempts=max(1, int(args.min_attempts)),
        max_units_cut=_clamp(float(args.max_units_cut), 0.0, 0.5),
        max_units_boost=_clamp(float(args.max_units_boost), 0.0, 0.3),
        max_prob_boost=_clamp(float(args.max_probability_boost), 0.0, 0.15),
    )
    _write_json_atomic(Path(args.output).resolve(), payload)
    print(
        f"[participation-allocator] wrote {Path(args.output).resolve()} "
        f"strategies={len(payload['strategies'])}"
    )


if __name__ == "__main__":
    main()
