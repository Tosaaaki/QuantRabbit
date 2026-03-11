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
from workers.common.setup_context import extract_setup_identity


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_json_loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _default_setup_min_attempts(min_attempts: int) -> int:
    return max(3, min(4, max(1, int(min_attempts))))


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


def _setup_realized_key(strategy_key: str, context: dict[str, Any]) -> str:
    payload = {"strategy_key": str(strategy_key or "").strip()}
    for key in ("setup_fingerprint", "flow_regime", "microstructure_bucket"):
        text = str(context.get(key) or "").strip()
        if text:
            payload[key] = text
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _build_allocation_record(
    record: dict[str, Any],
    *,
    realized_jpy: float,
    median_fill_rate: float,
    min_attempts: int,
    max_units_cut: float,
    max_units_boost: float,
    max_prob_boost: float,
) -> dict[str, Any]:
    attempts = int(record.get("attempts") or 0)
    fills = int(record.get("fills") or 0)
    filled_rate = _safe_float(record.get("filled_rate"), 0.0)
    attempt_share = _safe_float(record.get("attempt_share"), 0.0)
    fill_share = _safe_float(record.get("fill_share"), 0.0)
    share_gap = _safe_float(record.get("share_gap"), attempt_share - fill_share)
    units_multiplier = 1.0
    probability_multiplier = 1.0
    probability_offset = 0.0
    probability_boost = 0.0
    cadence_floor = 1.0
    action = "hold"
    terminal_status_counts = record.get("terminal_status_counts") if isinstance(record.get("terminal_status_counts"), dict) else {}
    hard_block_rate = 0.0
    probability_rejects = 0
    strategy_control_blocks = 0
    loss_drag_floor = max(8.0, float(max(1, min_attempts)) * 0.45)
    fast_lane_boost_attempts = 2
    small_sample_boost_attempts = max(3, min(int(min_attempts), 4))
    fast_winner_profit_floor = max(24.0, float(max(1, min_attempts)) * 4.0)
    if terminal_status_counts:
        hard_blocks = 0
        for status_name, count in terminal_status_counts.items():
            status_key = str(status_name or "").strip()
            if status_key == "entry_probability_reject":
                probability_rejects += int(count or 0)
            elif status_key == "strategy_control_entry_disabled":
                strategy_control_blocks += int(count or 0)
            if status_key in {"perf_block", "entry_probability_reject", "rejected"}:
                hard_blocks += int(count or 0)
        hard_block_rate = hard_blocks / max(1, attempts + hard_blocks)

    if attempts >= max(1, min_attempts):
        if share_gap >= 0.08 and filled_rate <= (median_fill_rate * 0.85) and realized_jpy <= 0.0:
            severity = _clamp((share_gap - 0.08) / 0.22, 0.0, 1.0)
            severity = max(severity, _clamp((median_fill_rate - filled_rate) / max(0.01, median_fill_rate), 0.0, 1.0))
            reject_pressure = _clamp((hard_block_rate - 0.20) / 0.55, 0.0, 1.0)
            loss_pressure = _clamp(abs(min(realized_jpy, 0.0)) / 240.0, 0.0, 1.0)
            trim_strength = max(
                severity,
                0.55 * severity + 0.30 * reject_pressure + 0.15 * loss_pressure,
            )
            units_multiplier = 1.0 - max_units_cut * (0.35 + 0.65 * severity)
            should_trim_probability = (
                loss_pressure >= _NEGATIVE_PROB_TRIM_MIN_LOSS_PRESSURE
                or (
                    severity >= _NEGATIVE_PROB_TRIM_MIN_SEVERITY
                    and reject_pressure >= _NEGATIVE_PROB_TRIM_MIN_REJECT_PRESSURE
                    and share_gap >= _NEGATIVE_PROB_TRIM_MIN_SHARE_GAP
                )
            )
            if should_trim_probability:
                probability_offset = -max_prob_boost * (0.20 + 0.80 * trim_strength)
            cadence_floor = 0.90
            action = "trim_units"
        elif (
            realized_jpy <= -loss_drag_floor
            and fills >= max(2, int(min_attempts * 0.5))
            and filled_rate >= max(median_fill_rate * 0.85, 0.10)
            and fill_share >= max(0.0, attempt_share - 0.04)
        ):
            loss_pressure = _clamp(abs(realized_jpy) / max(1.0, loss_drag_floor), 0.0, 1.0)
            participation_drag = _clamp((fill_share - attempt_share + 0.04) / 0.18, 0.0, 1.0)
            quality_drag = _clamp(
                (filled_rate / max(0.01, median_fill_rate) - 0.85) / 0.45,
                0.0,
                1.0,
            )
            trim_strength = max(
                loss_pressure,
                0.55 * loss_pressure + 0.25 * participation_drag + 0.20 * quality_drag,
            )
            units_multiplier = 1.0 - max_units_cut * (0.18 + 0.62 * trim_strength)
            should_trim_probability = (
                loss_pressure >= 0.45
                or (
                    fills >= max(4, int(min_attempts * 0.5))
                    and participation_drag >= 0.25
                )
            )
            if should_trim_probability:
                probability_offset = -max_prob_boost * (0.10 + 0.60 * trim_strength)
            cadence_floor = 0.94
            action = "trim_units"
        elif fill_share >= attempt_share + 0.02 and filled_rate >= median_fill_rate and realized_jpy >= 0.0:
            advantage = _clamp((fill_share - attempt_share) / 0.18, 0.0, 1.0)
            quality = _clamp((filled_rate - median_fill_rate) / max(0.01, median_fill_rate), 0.0, 1.0)
            boost_strength = max(advantage, quality)
            boost = max_units_boost * (0.42 + 0.58 * boost_strength)
            units_multiplier = 1.0 + boost
            probability_boost = max_prob_boost * (0.30 + 0.70 * boost_strength)
            cadence_floor = 1.0 + 0.22 * (0.35 + 0.65 * boost_strength)
            action = "boost_participation"
        elif filled_rate >= median_fill_rate * 1.10 and realized_jpy >= 0.0 and attempts >= max(min_attempts, 8):
            units_multiplier = 1.0 + max_units_boost * 0.52
            probability_boost = max_prob_boost * 0.42
            cadence_floor = 1.12
            action = "boost_participation"
    elif (
        attempts >= fast_lane_boost_attempts
        and fills >= 2
        and fills == attempts
        and realized_jpy > 0.0
        and filled_rate >= max(median_fill_rate * 1.05, 0.30)
        and fill_share >= attempt_share + 0.003
        and hard_block_rate <= 0.25
    ):
        participation_edge = _clamp((fill_share - attempt_share) / 0.14, 0.0, 1.0)
        sample_confidence = _clamp(
            (attempts - fast_lane_boost_attempts + 1) / max(1, int(min_attempts) - fast_lane_boost_attempts + 1),
            0.0,
            1.0,
        )
        fill_quality = _clamp(
            (filled_rate - max(0.20, median_fill_rate * 0.90))
            / max(0.05, 1.0 - max(0.20, median_fill_rate * 0.90)),
            0.0,
            1.0,
        )
        profit_confidence = _clamp(realized_jpy / fast_winner_profit_floor, 0.0, 1.0)
        boost_strength = max(
            participation_edge,
            fill_quality,
            0.25 * participation_edge + 0.15 * sample_confidence + 0.60 * profit_confidence,
        )
        units_multiplier = 1.0 + max_units_boost * (0.44 + 0.42 * boost_strength)
        probability_boost = max_prob_boost * (0.24 + 0.46 * boost_strength)
        cadence_floor = 1.0 + 0.09 + 0.11 * boost_strength
        action = "boost_participation"
    elif (
        attempts >= small_sample_boost_attempts
        and fills >= 2
        and realized_jpy > 0.0
        and filled_rate >= max(median_fill_rate * 1.05, 0.30)
        and fill_share >= attempt_share + 0.003
        and hard_block_rate <= 0.35
    ):
        participation_edge = _clamp((fill_share - attempt_share) / 0.12, 0.0, 1.0)
        sample_confidence = _clamp(
            (attempts - small_sample_boost_attempts + 1) / max(1, int(min_attempts) - small_sample_boost_attempts + 1),
            0.0,
            1.0,
        )
        profit_confidence = _clamp(realized_jpy / max(120.0, fast_winner_profit_floor * 1.5), 0.0, 1.0)
        boost_strength = max(
            participation_edge,
            0.45 * participation_edge + 0.20 * sample_confidence + 0.35 * profit_confidence,
        )
        units_multiplier = 1.0 + max_units_boost * (0.38 + 0.52 * boost_strength)
        probability_boost = max_prob_boost * (0.22 + 0.56 * boost_strength)
        cadence_floor = 1.0 + 0.08 + 0.11 * boost_strength
        action = "boost_participation"
    elif (
        attempts >= 1
        and fills >= 1
        and realized_jpy > 0.0
        and probability_rejects >= max(8, fills * 8)
        and strategy_control_blocks <= probability_rejects
        and fill_share >= attempt_share + 0.001
    ):
        reject_pressure = _clamp(probability_rejects / 64.0, 0.0, 1.0)
        profit_confidence = _clamp(realized_jpy / 80.0, 0.0, 1.0)
        boost_strength = max(0.25, 0.65 * reject_pressure + 0.35 * profit_confidence)
        units_multiplier = 1.0 + max_units_boost * (0.08 + 0.14 * boost_strength)
        probability_boost = max_prob_boost * (0.12 + 0.23 * boost_strength)
        cadence_floor = 1.0 + 0.04 + 0.04 * boost_strength
        action = "boost_participation"

    quality_score = _clamp(
        0.45 * _clamp(filled_rate / max(0.01, median_fill_rate), 0.0, 1.25)
        + 0.30 * _clamp(fill_share / max(0.01, attempt_share) if attempt_share > 0 else 0.0, 0.0, 1.25)
        + 0.25 * _clamp((realized_jpy + 500.0) / 1500.0, 0.0, 1.0),
        0.0,
        1.25,
    )

    return {
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
        "probability_multiplier": round(_clamp(probability_multiplier, 0.75, 1.25), 4),
        "probability_offset": round(_clamp(probability_offset, -max_prob_boost, max_prob_boost), 4),
        "probability_boost": round(_clamp(probability_boost, 0.0, max_prob_boost), 4),
        "cadence_floor": round(_clamp(cadence_floor, 0.85, 1.24), 4),
        "quality_score": round(quality_score, 4),
        "hard_block_rate": round(hard_block_rate, 6),
        "max_units_cut": round(float(max_units_cut), 4),
        "max_units_boost": round(float(max_units_boost), 4),
        "max_probability_boost": round(float(max_prob_boost), 4),
        "action": action,
    }


def _build_setup_overrides(
    strategy_key: str,
    setups: Any,
    *,
    realized_by_setup: dict[str, float],
    median_fill_rate: float,
    min_attempts: int,
    setup_min_attempts: int | None,
    max_units_cut: float,
    max_units_boost: float,
    max_prob_boost: float,
) -> list[dict[str, Any]]:
    if isinstance(setups, dict):
        iterable = list(setups.values())
    elif isinstance(setups, list):
        iterable = list(setups)
    else:
        return []
    resolved_setup_min_attempts = max(
        1,
        int(
            setup_min_attempts
            if setup_min_attempts is not None
            else _default_setup_min_attempts(min_attempts)
        ),
    )
    setup_fill_rates = [
        _safe_float(item.get("filled_rate"), 0.0)
        for item in iterable
        if isinstance(item, dict)
        and int(item.get("attempts") or 0) >= resolved_setup_min_attempts
    ]
    setup_median_fill_rate = _median(setup_fill_rates)
    if setup_median_fill_rate <= 0.0:
        setup_median_fill_rate = median_fill_rate
    overrides: list[dict[str, Any]] = []
    for setup in iterable:
        if not isinstance(setup, dict):
            continue
        context = {
            "setup_fingerprint": str(setup.get("setup_fingerprint") or "").strip(),
            "flow_regime": str(setup.get("flow_regime") or "").strip(),
            "microstructure_bucket": str(setup.get("microstructure_bucket") or "").strip(),
        }
        if not any(context.values()):
            continue
        realized_jpy = _safe_float(
            realized_by_setup.get(_setup_realized_key(strategy_key, context)),
            0.0,
        )
        metrics = _build_allocation_record(
            setup,
            realized_jpy=realized_jpy,
            median_fill_rate=setup_median_fill_rate,
            min_attempts=resolved_setup_min_attempts,
            max_units_cut=max_units_cut,
            max_units_boost=max_units_boost,
            max_prob_boost=max_prob_boost,
        )
        if (
            str(metrics.get("action") or "hold") == "hold"
            and float(metrics.get("probability_offset") or 0.0) == 0.0
            and float(metrics.get("probability_boost") or 0.0) == 0.0
            and abs(float(metrics.get("lot_multiplier") or 1.0) - 1.0) < 1e-9
        ):
            continue
        overrides.append(
            {
                "match_dimension": (
                    str(setup.get("match_dimension") or "").strip()
                    or (
                        "setup_fingerprint"
                        if context["setup_fingerprint"]
                        else "flow_micro"
                        if context["flow_regime"] and context["microstructure_bucket"]
                        else "flow_regime"
                        if context["flow_regime"]
                        else "microstructure_bucket"
                        if context["microstructure_bucket"]
                        else "unknown"
                    )
                ),
                "setup_fingerprint": context["setup_fingerprint"],
                "flow_regime": context["flow_regime"],
                "microstructure_bucket": context["microstructure_bucket"],
                **metrics,
            }
        )
    overrides.sort(
        key=lambda item: (
            {
                "setup_fingerprint": 4,
                "flow_micro": 3,
                "flow_regime": 2,
                "microstructure_bucket": 1,
            }.get(str(item.get("match_dimension") or ""), 0),
            int(item.get("attempts") or 0),
        ),
        reverse=True,
    )
    return overrides


_NEGATIVE_PROB_TRIM_MIN_LOSS_PRESSURE = 0.08
_NEGATIVE_PROB_TRIM_MIN_SEVERITY = 0.78
_NEGATIVE_PROB_TRIM_MIN_REJECT_PRESSURE = 0.80
_NEGATIVE_PROB_TRIM_MIN_SHARE_GAP = 0.18


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


def _hard_block_metrics(record: dict[str, Any], attempts: int) -> tuple[float, int, int]:
    terminal_status_counts = (
        record.get("terminal_status_counts")
        if isinstance(record.get("terminal_status_counts"), dict)
        else {}
    )
    hard_block_rate = 0.0
    probability_rejects = 0
    strategy_control_blocks = 0
    if terminal_status_counts:
        hard_blocks = 0
        for status_name, count in terminal_status_counts.items():
            status_key = str(status_name or "").strip()
            if status_key == "entry_probability_reject":
                probability_rejects += int(count or 0)
            elif status_key == "strategy_control_entry_disabled":
                strategy_control_blocks += int(count or 0)
            if status_key in {"perf_block", "entry_probability_reject", "rejected"}:
                hard_blocks += int(count or 0)
        hard_block_rate = hard_blocks / max(1, attempts + hard_blocks)
    return hard_block_rate, probability_rejects, strategy_control_blocks


def _allocation_record(
    record: dict[str, Any],
    *,
    realized_jpy: float,
    median_fill_rate: float,
    min_attempts: int,
    small_sample_boost_attempts: int,
    max_units_cut: float,
    max_units_boost: float,
    max_prob_boost: float,
) -> dict[str, Any]:
    attempts = int(record.get("attempts") or 0)
    fills = int(record.get("fills") or 0)
    filled_rate = _safe_float(record.get("filled_rate"), 0.0)
    attempt_share = _safe_float(record.get("attempt_share"), 0.0)
    fill_share = _safe_float(record.get("fill_share"), 0.0)
    share_gap = _safe_float(record.get("share_gap"), attempt_share - fill_share)
    units_multiplier = 1.0
    probability_multiplier = 1.0
    probability_offset = 0.0
    probability_boost = 0.0
    cadence_floor = 1.0
    action = "hold"
    hard_block_rate, probability_rejects, strategy_control_blocks = _hard_block_metrics(record, attempts)

    if attempts >= max(1, min_attempts):
        if share_gap >= 0.08 and filled_rate <= (median_fill_rate * 0.85) and realized_jpy <= 0.0:
            severity = _clamp((share_gap - 0.08) / 0.22, 0.0, 1.0)
            severity = max(
                severity,
                _clamp((median_fill_rate - filled_rate) / max(0.01, median_fill_rate), 0.0, 1.0),
            )
            reject_pressure = _clamp((hard_block_rate - 0.20) / 0.55, 0.0, 1.0)
            loss_pressure = _clamp(abs(min(realized_jpy, 0.0)) / 240.0, 0.0, 1.0)
            trim_strength = max(
                severity,
                0.55 * severity + 0.30 * reject_pressure + 0.15 * loss_pressure,
            )
            units_multiplier = 1.0 - max_units_cut * (0.35 + 0.65 * severity)
            should_trim_probability = (
                loss_pressure >= _NEGATIVE_PROB_TRIM_MIN_LOSS_PRESSURE
                or (
                    severity >= _NEGATIVE_PROB_TRIM_MIN_SEVERITY
                    and reject_pressure >= _NEGATIVE_PROB_TRIM_MIN_REJECT_PRESSURE
                    and share_gap >= _NEGATIVE_PROB_TRIM_MIN_SHARE_GAP
                )
            )
            if should_trim_probability:
                probability_offset = -max_prob_boost * (0.20 + 0.80 * trim_strength)
            cadence_floor = 0.90
            action = "trim_units"
        elif fill_share >= attempt_share + 0.02 and filled_rate >= median_fill_rate and realized_jpy >= 0.0:
            advantage = _clamp((fill_share - attempt_share) / 0.18, 0.0, 1.0)
            quality = _clamp((filled_rate - median_fill_rate) / max(0.01, median_fill_rate), 0.0, 1.0)
            boost_strength = max(advantage, quality)
            boost = max_units_boost * (0.30 + 0.70 * boost_strength)
            units_multiplier = 1.0 + boost
            probability_boost = max_prob_boost * (0.25 + 0.75 * boost_strength)
            cadence_floor = 1.0 + 0.18 * (0.30 + 0.70 * boost_strength)
            action = "boost_participation"
        elif filled_rate >= median_fill_rate * 1.10 and realized_jpy >= 0.0 and attempts >= max(min_attempts, 8):
            units_multiplier = 1.0 + max_units_boost * 0.40
            probability_boost = max_prob_boost * 0.35
            cadence_floor = 1.08
            action = "boost_participation"
    elif (
        attempts >= small_sample_boost_attempts
        and fills >= 2
        and realized_jpy > 0.0
        and filled_rate >= max(median_fill_rate * 1.05, 0.30)
        and fill_share >= attempt_share + 0.003
        and hard_block_rate <= 0.35
    ):
        participation_edge = _clamp((fill_share - attempt_share) / 0.12, 0.0, 1.0)
        sample_confidence = _clamp(
            (attempts - small_sample_boost_attempts + 1)
            / max(1, int(min_attempts) - small_sample_boost_attempts + 1),
            0.0,
            1.0,
        )
        profit_confidence = _clamp(realized_jpy / 400.0, 0.0, 1.0)
        boost_strength = max(
            participation_edge,
            0.55 * participation_edge + 0.25 * sample_confidence + 0.20 * profit_confidence,
        )
        units_multiplier = 1.0 + max_units_boost * (0.18 + 0.32 * boost_strength)
        probability_boost = max_prob_boost * (0.12 + 0.28 * boost_strength)
        cadence_floor = 1.0 + 0.06 + 0.08 * boost_strength
        action = "boost_participation"
    elif (
        attempts >= 1
        and fills >= 1
        and realized_jpy > 0.0
        and probability_rejects >= max(8, fills * 8)
        and strategy_control_blocks <= probability_rejects
        and fill_share >= attempt_share + 0.001
    ):
        reject_pressure = _clamp(probability_rejects / 64.0, 0.0, 1.0)
        profit_confidence = _clamp(realized_jpy / 80.0, 0.0, 1.0)
        boost_strength = max(0.25, 0.65 * reject_pressure + 0.35 * profit_confidence)
        units_multiplier = 1.0 + max_units_boost * (0.04 + 0.08 * boost_strength)
        probability_boost = max_prob_boost * (0.08 + 0.17 * boost_strength)
        cadence_floor = 1.0 + 0.03 + 0.03 * boost_strength
        action = "boost_participation"

    quality_score = _clamp(
        0.45 * _clamp(filled_rate / max(0.01, median_fill_rate), 0.0, 1.25)
        + 0.30 * _clamp(fill_share / max(0.01, attempt_share) if attempt_share > 0 else 0.0, 0.0, 1.25)
        + 0.25 * _clamp((realized_jpy + 500.0) / 1500.0, 0.0, 1.0),
        0.0,
        1.25,
    )

    return {
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
        "probability_multiplier": round(_clamp(probability_multiplier, 0.75, 1.25), 4),
        "probability_offset": round(_clamp(probability_offset, -max_prob_boost, max_prob_boost), 4),
        "probability_boost": round(_clamp(probability_boost, 0.0, max_prob_boost), 4),
        "cadence_floor": round(_clamp(cadence_floor, 0.85, 1.18), 4),
        "quality_score": round(quality_score, 4),
        "hard_block_rate": round(hard_block_rate, 6),
        "action": action,
    }


def _build_setup_overrides_legacy(
    record: dict[str, Any],
    *,
    realized_jpy: float,
    median_fill_rate: float,
    min_attempts: int,
    max_units_cut: float,
    max_units_boost: float,
    max_prob_boost: float,
) -> list[dict[str, Any]]:
    setups = record.get("setups")
    if not isinstance(setups, list):
        return []
    setup_min_attempts = max(6, min(int(min_attempts), max(6, int(round(min_attempts * 0.5)))))
    setup_fill_rates = [
        _safe_float(item.get("filled_rate"), 0.0)
        for item in setups
        if isinstance(item, dict) and int(item.get("attempts") or 0) >= setup_min_attempts
    ]
    setup_median_fill_rate = _median(setup_fill_rates)
    if setup_median_fill_rate <= 0.0:
        setup_median_fill_rate = median_fill_rate
    setup_small_sample_boost_attempts = max(2, min(setup_min_attempts, 4))

    overrides: list[dict[str, Any]] = []
    for item in setups:
        if not isinstance(item, dict):
            continue
        metrics = _allocation_record(
            item,
            realized_jpy=realized_jpy,
            median_fill_rate=setup_median_fill_rate,
            min_attempts=setup_min_attempts,
            small_sample_boost_attempts=setup_small_sample_boost_attempts,
            max_units_cut=max_units_cut,
            max_units_boost=max_units_boost,
            max_prob_boost=max_prob_boost,
        )
        changed = (
            metrics["action"] != "hold"
            or abs(float(metrics["lot_multiplier"]) - 1.0) > 1e-9
            or abs(float(metrics["probability_offset"])) > 1e-9
            or abs(float(metrics["probability_boost"])) > 1e-9
        )
        if not changed:
            continue
        overrides.append(
            {
                "match_dimension": str(item.get("match_dimension") or "setup_fingerprint"),
                "setup_fingerprint": str(item.get("setup_fingerprint") or "") or None,
                "flow_regime": str(item.get("flow_regime") or "") or None,
                "microstructure_bucket": str(item.get("microstructure_bucket") or "") or None,
                **metrics,
            }
        )
    overrides.sort(
        key=lambda item: (
            {
                "setup_fingerprint": 4,
                "flow_micro": 3,
                "flow_regime": 2,
                "microstructure_bucket": 1,
            }.get(str(item.get("match_dimension") or ""), 0),
            int(item.get("attempts") or 0),
        ),
        reverse=True,
    )
    return overrides


def _load_recent_realized_setup_jpy(trades_db: Path, *, lookback_hours: float) -> dict[str, float]:
    if not trades_db.exists():
        return {}
    out: dict[str, float] = {}
    con = sqlite3.connect(f"file:{trades_db}?mode=ro", uri=True, timeout=8.0, isolation_level=None)
    try:
        columns = {
            str(row[1])
            for row in con.execute("PRAGMA table_info(trades)").fetchall()
            if row[1] is not None
        }
        if "entry_thesis" not in columns:
            return {}
        units_select = "units" if "units" in columns else "0 AS units"
        cur = con.cursor()
        cur.execute(
            f"""
            SELECT strategy_tag, strategy, entry_thesis, {units_select}, COALESCE(realized_pl, 0.0)
            FROM trades
            WHERE close_time IS NOT NULL
              AND julianday(close_time) >= julianday('now', ?)
            """,
            (f"-{max(0.1, float(lookback_hours)):.3f} hours",),
        )
        for strategy_tag, strategy, entry_thesis, units, realized_pl in cur.fetchall():
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
            context = extract_setup_identity(_safe_json_loads(entry_thesis), units=_safe_int(units, 0))
            if not context:
                continue
            realized_key = _setup_realized_key(strategy_key, context)
            out[realized_key] = out.get(realized_key, 0.0) + _safe_float(realized_pl, 0.0)
    finally:
        con.close()
    return out


def build_participation_alloc(
    entry_path_summary: dict[str, Any],
    *,
    realized_by_strategy: dict[str, float],
    realized_by_setup: dict[str, float] | None = None,
    min_attempts: int,
    setup_min_attempts: int | None = None,
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
    realized_by_setup = realized_by_setup if isinstance(realized_by_setup, dict) else {}
    resolved_setup_min_attempts = max(
        1,
        int(
            setup_min_attempts
            if setup_min_attempts is not None
            else _default_setup_min_attempts(min_attempts)
        ),
    )

    output_strategies: dict[str, Any] = {}
    for raw_key, record in sorted(strategies.items()):
        if not isinstance(record, dict):
            continue
        strategy_key = resolve_strategy_tag(str(raw_key or "").strip()) or str(raw_key or "").strip()
        realized_jpy = _safe_float(realized_by_strategy.get(strategy_key), 0.0)
        output_record = {
            "strategy_key": str(strategy_key or raw_key),
            "pocket": str(record.get("pocket") or "").strip() or "unknown",
            **_build_allocation_record(
                record,
                realized_jpy=realized_jpy,
                median_fill_rate=median_fill_rate,
                min_attempts=min_attempts,
                max_units_cut=max_units_cut,
                max_units_boost=max_units_boost,
                max_prob_boost=max_prob_boost,
            ),
        }
        setup_overrides = _build_setup_overrides(
            str(strategy_key or raw_key),
            record.get("setups"),
            realized_by_setup=realized_by_setup,
            median_fill_rate=median_fill_rate,
            min_attempts=min_attempts,
            setup_min_attempts=resolved_setup_min_attempts,
            max_units_cut=max_units_cut,
            max_units_boost=max_units_boost,
            max_prob_boost=max_prob_boost,
        )
        if setup_overrides:
            output_record["setup_overrides"] = setup_overrides
        output_strategies[str(strategy_key or raw_key)] = output_record

    action_counts: dict[str, int] = {}
    for rec in output_strategies.values():
        action = str(rec.get("action") or "hold")
        action_counts[action] = action_counts.get(action, 0) + 1
    negative_probability_offsets_enabled = any(
        float(rec.get("probability_offset") or 0.0) < 0.0 for rec in output_strategies.values()
    ) or any(
        float(override.get("probability_offset") or 0.0) < 0.0
        for rec in output_strategies.values()
        for override in (rec.get("setup_overrides") or [])
    )

    return {
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "lookback_hours": _safe_float(entry_path_summary.get("lookback_hours"), 24.0),
        "median_filled_rate": round(median_fill_rate, 6),
        "allocation_policy": {
            "protect_frequency": True,
            "min_attempts": int(min_attempts),
            "setup_min_attempts": int(resolved_setup_min_attempts),
            "max_units_cut": round(max_units_cut, 4),
            "max_units_boost": round(max_units_boost, 4),
            "max_probability_boost": round(max_prob_boost, 4),
            "negative_probability_offsets_enabled": negative_probability_offsets_enabled,
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
    ap.add_argument("--setup-min-attempts", type=int, default=4)
    ap.add_argument("--max-units-cut", type=float, default=0.18)
    ap.add_argument("--max-units-boost", type=float, default=0.18)
    ap.add_argument("--max-probability-boost", type=float, default=0.08)
    args = ap.parse_args()

    summary = _read_json(Path(args.entry_path_summary).resolve())
    realized_by_strategy = _load_recent_realized_jpy(
        Path(args.trades_db).resolve(),
        lookback_hours=float(args.lookback_hours),
    )
    realized_by_setup = _load_recent_realized_setup_jpy(
        Path(args.trades_db).resolve(),
        lookback_hours=float(args.lookback_hours),
    )
    payload = build_participation_alloc(
        summary,
        realized_by_strategy=realized_by_strategy,
        realized_by_setup=realized_by_setup,
        min_attempts=max(1, int(args.min_attempts)),
        setup_min_attempts=max(1, int(args.setup_min_attempts)),
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
