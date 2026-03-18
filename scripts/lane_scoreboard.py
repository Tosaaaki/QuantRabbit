#!/usr/bin/env python3
"""Build lane-level winner/loser gates from recent entry-path and trade outcomes."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
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
        return dict(raw)
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


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _strategy_key(
    *,
    strategy_tag: Any,
    strategy: Any,
    entry_thesis: Any,
) -> str:
    raw_key, _canonical_key = extract_strategy_tags(
        strategy_tag=strategy_tag,
        strategy=strategy,
        entry_thesis=entry_thesis,
    )
    if raw_key:
        return raw_key
    resolved = resolve_strategy_tag(str(strategy_tag or strategy or "").strip())
    return resolved or str(strategy_tag or strategy or "").strip()


def _setup_key(strategy_key: str, context: dict[str, Any]) -> str:
    payload = {"strategy_key": str(strategy_key or "").strip()}
    for key in ("setup_fingerprint", "flow_regime", "microstructure_bucket"):
        text = str(context.get(key) or "").strip()
        if text:
            payload[key] = text
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _empty_trade_bucket() -> dict[str, Any]:
    return {
        "closed_trades": 0,
        "wins": 0,
        "losses": 0,
        "sum_realized_jpy": 0.0,
        "sum_pips": 0.0,
        "gross_profit_jpy": 0.0,
        "gross_loss_jpy": 0.0,
        "stop_loss_count": 0,
        "margin_closeout_count": 0,
        "market_close_loss_count": 0,
        "close_reason_counts": Counter(),
    }


def _update_trade_bucket(
    bucket: dict[str, Any],
    *,
    realized_jpy: float,
    pl_pips: float,
    close_reason: str,
) -> None:
    bucket["closed_trades"] += 1
    bucket["sum_realized_jpy"] += realized_jpy
    bucket["sum_pips"] += pl_pips
    if realized_jpy > 0.0:
        bucket["wins"] += 1
        bucket["gross_profit_jpy"] += realized_jpy
    elif realized_jpy < 0.0:
        bucket["losses"] += 1
        bucket["gross_loss_jpy"] += abs(realized_jpy)
    if close_reason == "STOP_LOSS_ORDER":
        bucket["stop_loss_count"] += 1
    elif close_reason == "MARKET_ORDER_MARGIN_CLOSEOUT":
        bucket["margin_closeout_count"] += 1
    elif close_reason == "MARKET_ORDER_TRADE_CLOSE" and realized_jpy < 0.0:
        bucket["market_close_loss_count"] += 1
    counts = bucket.get("close_reason_counts")
    if isinstance(counts, Counter):
        counts[close_reason or "unknown"] += 1


def _finalize_trade_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    closed_trades = int(bucket.get("closed_trades") or 0)
    wins = int(bucket.get("wins") or 0)
    losses = int(bucket.get("losses") or 0)
    gross_profit_jpy = _safe_float(bucket.get("gross_profit_jpy"), 0.0)
    gross_loss_jpy = _safe_float(bucket.get("gross_loss_jpy"), 0.0)
    realized_jpy = _safe_float(bucket.get("sum_realized_jpy"), 0.0)
    sum_pips = _safe_float(bucket.get("sum_pips"), 0.0)
    stop_loss_like_count = _safe_int(bucket.get("stop_loss_count"), 0) + _safe_int(
        bucket.get("margin_closeout_count"),
        0,
    )
    if gross_loss_jpy > 0.0:
        profit_factor = gross_profit_jpy / gross_loss_jpy
    elif gross_profit_jpy > 0.0:
        profit_factor = gross_profit_jpy
    else:
        profit_factor = 0.0
    close_reason_counts = bucket.get("close_reason_counts")
    if isinstance(close_reason_counts, Counter) and close_reason_counts:
        primary_close_reason = close_reason_counts.most_common(1)[0][0]
        close_reason_map = {
            str(key): int(value)
            for key, value in sorted(
                close_reason_counts.items(), key=lambda item: (-item[1], item[0])
            )
        }
    else:
        primary_close_reason = ""
        close_reason_map = {}
    return {
        "closed_trades": closed_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / float(max(1, closed_trades)), 6),
        "realized_jpy": round(realized_jpy, 3),
        "sum_pips": round(sum_pips, 3),
        "gross_profit_jpy": round(gross_profit_jpy, 3),
        "gross_loss_jpy": round(gross_loss_jpy, 3),
        "profit_factor": round(min(profit_factor, 9.99), 6),
        "avg_realized_jpy": round(realized_jpy / float(max(1, closed_trades)), 6),
        "avg_pips": round(sum_pips / float(max(1, closed_trades)), 6),
        "stop_loss_count": _safe_int(bucket.get("stop_loss_count"), 0),
        "margin_closeout_count": _safe_int(bucket.get("margin_closeout_count"), 0),
        "market_close_loss_count": _safe_int(bucket.get("market_close_loss_count"), 0),
        "stop_loss_like_count": stop_loss_like_count,
        "stop_loss_rate": round(stop_loss_like_count / float(max(1, closed_trades)), 6),
        "primary_close_reason": primary_close_reason,
        "close_reason_counts": close_reason_map,
    }


def _load_trade_metrics_by_setup(
    trades_db: Path,
    *,
    lookback_hours: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    if not trades_db.exists():
        return {}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    con = sqlite3.connect(
        f"file:{trades_db}?mode=ro",
        uri=True,
        timeout=8.0,
        isolation_level=None,
    )
    try:
        columns = {
            str(row[1])
            for row in con.execute("PRAGMA table_info(trades)").fetchall()
            if row[1] is not None
        }
        if "entry_thesis" not in columns:
            return {}
        units_select = "units" if "units" in columns else "0 AS units"
        pl_pips_select = "pl_pips" if "pl_pips" in columns else "0.0 AS pl_pips"
        cur = con.cursor()
        cur.execute(
            f"""
            SELECT strategy_tag, strategy, entry_thesis, {units_select},
                   COALESCE(realized_pl, 0.0), COALESCE({pl_pips_select}, 0.0),
                   COALESCE(close_reason, '')
            FROM trades
            WHERE close_time IS NOT NULL
              AND julianday(close_time) >= julianday('now', ?)
            ORDER BY julianday(close_time) DESC
            """,
            (f"-{max(0.1, float(lookback_hours)):.3f} hours",),
        )
        for (
            strategy_tag,
            strategy,
            entry_thesis_raw,
            units,
            realized_pl,
            pl_pips,
            close_reason,
        ) in cur.fetchall():
            entry_thesis = _safe_json_loads(entry_thesis_raw)
            strategy_key = _strategy_key(
                strategy_tag=strategy_tag,
                strategy=strategy,
                entry_thesis=entry_thesis,
            )
            if not strategy_key:
                continue
            setup_context = extract_setup_identity(
                entry_thesis, units=_safe_int(units, 0)
            )
            if not setup_context:
                continue
            setup_key = _setup_key(strategy_key, setup_context)
            bucket = out.setdefault(strategy_key, {}).setdefault(
                setup_key, _empty_trade_bucket()
            )
            _update_trade_bucket(
                bucket,
                realized_jpy=_safe_float(realized_pl, 0.0),
                pl_pips=_safe_float(pl_pips, 0.0),
                close_reason=str(close_reason or "").strip() or "unknown",
            )
    finally:
        con.close()
    return {
        strategy_key: {
            setup_key: _finalize_trade_bucket(bucket)
            for setup_key, bucket in setup_buckets.items()
        }
        for strategy_key, setup_buckets in out.items()
    }


def _quality_score(
    *,
    filled_rate: float,
    median_filled_rate: float,
    attempt_share: float,
    fill_share: float,
    realized_jpy: float,
    profit_factor: float,
    win_rate: float,
) -> float:
    fill_quality = _clamp(
        filled_rate / max(0.02, median_filled_rate),
        0.0,
        1.25,
    )
    share_quality = _clamp(
        fill_share / max(0.01, attempt_share) if attempt_share > 0.0 else 0.0,
        0.0,
        1.25,
    )
    realized_quality = _clamp((realized_jpy + 80.0) / 240.0, 0.0, 1.0)
    pf_quality = _clamp((profit_factor - 0.80) / 0.80, 0.0, 1.0)
    return round(
        _clamp(
            0.30 * fill_quality
            + 0.22 * share_quality
            + 0.22 * realized_quality
            + 0.16 * pf_quality
            + 0.10 * win_rate,
            0.0,
            1.25,
        ),
        4,
    )


def _promotion_gate(
    *,
    attempts: int,
    fills: int,
    filled_rate: float,
    median_filled_rate: float,
    attempt_share: float,
    fill_share: float,
    hard_block_rate: float,
    trade_metrics: dict[str, Any],
    min_attempts: int,
    max_units_boost: float,
    max_prob_boost: float,
) -> dict[str, Any]:
    closed_trades = _safe_int(trade_metrics.get("closed_trades"), 0)
    realized_jpy = _safe_float(trade_metrics.get("realized_jpy"), 0.0)
    win_rate = _safe_float(trade_metrics.get("win_rate"), 0.0)
    profit_factor = _safe_float(trade_metrics.get("profit_factor"), 0.0)
    stop_loss_rate = _safe_float(trade_metrics.get("stop_loss_rate"), 0.0)
    fill_edge = fill_share - attempt_share
    reasons: list[str] = []
    checks = {
        "enough_attempts": attempts >= max(2, min_attempts),
        "enough_fills": fills >= 2,
        "enough_closed_trades": closed_trades >= 2,
        "positive_realized": realized_jpy > 0.0,
        "healthy_pf": profit_factor >= 1.02
        or (closed_trades > 0 and _safe_int(trade_metrics.get("losses"), 0) == 0),
        "healthy_win_rate": win_rate >= 0.50
        or (
            closed_trades > 0
            and _safe_int(trade_metrics.get("wins"), 0) == closed_trades
        ),
        "limited_stop_loss": stop_loss_rate <= 0.45,
        "limited_hard_blocks": hard_block_rate <= 0.35,
        "healthy_fill_rate": filled_rate >= max(median_filled_rate * 1.05, 0.25),
        "fill_edge": fill_edge >= 0.003,
    }
    for reason, passed in checks.items():
        if passed:
            reasons.append(reason)
    passed = all(checks.values())
    fill_edge_strength = _clamp(fill_edge / 0.14, 0.0, 1.0)
    fill_quality = _clamp(
        (filled_rate - max(0.20, median_filled_rate * 0.90))
        / max(0.05, 1.0 - max(0.20, median_filled_rate * 0.90)),
        0.0,
        1.0,
    )
    sample_confidence = _clamp(
        (min(attempts, max(fills, closed_trades)) - max(2, min_attempts) + 1)
        / max(1, max(2, min_attempts)),
        0.0,
        1.0,
    )
    profit_confidence = _clamp(
        realized_jpy / max(24.0, float(max(1, min_attempts)) * 6.0), 0.0, 1.0
    )
    pf_confidence = _clamp((profit_factor - 1.0) / 0.60, 0.0, 1.0)
    strength = max(
        fill_edge_strength,
        0.18 * sample_confidence
        + 0.26 * fill_quality
        + 0.30 * profit_confidence
        + 0.26 * pf_confidence,
    )
    units_multiplier = round(
        1.0 + max_units_boost * (0.36 + 0.54 * strength),
        4,
    )
    probability_boost = round(
        max_prob_boost * (0.20 + 0.55 * strength),
        4,
    )
    cadence_floor = round(1.0 + 0.08 + 0.10 * strength, 4)
    return {
        "passed": passed,
        "strength": round(_clamp(strength, 0.0, 1.0), 4),
        "reasons": reasons,
        "units_multiplier": units_multiplier,
        "lot_multiplier": units_multiplier,
        "probability_boost": probability_boost,
        "probability_offset": 0.0,
        "max_probability_cut": 0.0,
        "cadence_floor": cadence_floor,
    }


def _quarantine_gate(
    *,
    attempts: int,
    fills: int,
    filled_rate: float,
    median_filled_rate: float,
    share_gap: float,
    hard_block_rate: float,
    trade_metrics: dict[str, Any],
    min_attempts: int,
    max_units_cut: float,
    max_prob_boost: float,
) -> dict[str, Any]:
    closed_trades = _safe_int(trade_metrics.get("closed_trades"), 0)
    realized_jpy = _safe_float(trade_metrics.get("realized_jpy"), 0.0)
    avg_realized_jpy = _safe_float(trade_metrics.get("avg_realized_jpy"), 0.0)
    win_rate = _safe_float(trade_metrics.get("win_rate"), 0.0)
    profit_factor = _safe_float(trade_metrics.get("profit_factor"), 0.0)
    stop_loss_rate = _safe_float(trade_metrics.get("stop_loss_rate"), 0.0)
    stop_loss_like_count = _safe_int(trade_metrics.get("stop_loss_like_count"), 0)
    severe_fresh_loser = bool(
        attempts >= 2
        and fills >= 2
        and closed_trades >= 2
        and realized_jpy <= -8.0
        and stop_loss_like_count >= 1
        and (win_rate <= 0.25 or profit_factor <= 0.75)
    )
    chronic_loser = bool(
        attempts >= max(2, min_attempts)
        and closed_trades >= 2
        and realized_jpy < 0.0
        and (
            stop_loss_rate >= 0.50
            or (profit_factor <= 0.90 and win_rate <= 0.35)
            or avg_realized_jpy <= -4.0
            or (
                share_gap >= 0.08
                and hard_block_rate >= 0.20
                and filled_rate <= max(0.02, median_filled_rate * 0.95)
            )
        )
    )
    active = severe_fresh_loser or chronic_loser
    reasons: list[str] = []
    if stop_loss_rate >= 0.50:
        reasons.append("high_stop_loss_rate")
    if realized_jpy < 0.0:
        reasons.append("negative_realized_jpy")
    if profit_factor <= 0.90 and closed_trades > 0:
        reasons.append("weak_profit_factor")
    if share_gap >= 0.08:
        reasons.append("overused_share_gap")
    if hard_block_rate >= 0.20:
        reasons.append("hard_block_drag")
    if severe_fresh_loser:
        reasons.append("fresh_loser_burst")
    loss_pressure = _clamp(
        abs(min(realized_jpy, 0.0)) / max(12.0, float(max(1, min_attempts)) * 6.0),
        0.0,
        1.0,
    )
    stop_pressure = _clamp((stop_loss_rate - 0.30) / 0.50, 0.0, 1.0)
    pf_deficit = _clamp((1.0 - min(profit_factor, 1.0)) / 0.60, 0.0, 1.0)
    share_pressure = _clamp((share_gap - 0.05) / 0.25, 0.0, 1.0)
    block_pressure = _clamp((hard_block_rate - 0.15) / 0.50, 0.0, 1.0)
    severity = max(
        loss_pressure,
        0.28 * loss_pressure
        + 0.28 * stop_pressure
        + 0.18 * pf_deficit
        + 0.14 * share_pressure
        + 0.12 * block_pressure,
    )
    units_multiplier = round(
        1.0 - max_units_cut * (0.28 + 0.58 * severity),
        4,
    )
    max_probability_cut = 0.0
    probability_offset = 0.0
    if stop_loss_rate >= 0.50 or realized_jpy <= -20.0 or severity >= 0.55:
        max_probability_cut = min(
            0.25,
            max(max_prob_boost, max_prob_boost * (0.75 + 0.75 * severity)),
        )
        probability_offset = -max_probability_cut * (0.10 + 0.60 * severity)
    cadence_floor = round(0.96 - 0.08 * severity, 4)
    return {
        "active": active,
        "severity": round(_clamp(severity, 0.0, 1.0), 4),
        "reasons": reasons,
        "units_multiplier": round(units_multiplier, 4),
        "lot_multiplier": round(units_multiplier, 4),
        "probability_boost": 0.0,
        "probability_offset": round(probability_offset, 4),
        "max_probability_cut": round(max_probability_cut, 4),
        "cadence_floor": cadence_floor,
    }


def build_lane_scoreboard(
    entry_path_summary: dict[str, Any],
    *,
    trade_metrics_by_setup: dict[str, dict[str, dict[str, Any]]],
    min_attempts: int,
    setup_min_attempts: int,
    max_units_cut: float,
    max_units_boost: float,
    max_prob_boost: float,
) -> dict[str, Any]:
    strategies = entry_path_summary.get("strategies")
    if not isinstance(strategies, dict):
        strategies = {}

    strategy_rows: dict[str, Any] = {}
    strategy_count = 0
    lane_count = 0
    action_counts = {"boost_participation": 0, "trim_units": 0, "hold": 0}

    for raw_key, record in sorted(strategies.items()):
        if not isinstance(record, dict):
            continue
        strategy_key = (
            resolve_strategy_tag(str(raw_key or "").strip())
            or str(raw_key or "").strip()
        )
        setups_raw = record.get("setups")
        if not isinstance(setups_raw, list) or not setups_raw:
            continue
        strategy_count += 1
        setup_fill_rates = [
            _safe_float(item.get("filled_rate"), 0.0)
            for item in setups_raw
            if isinstance(item, dict)
            and _safe_int(item.get("attempts"), 0) >= max(1, setup_min_attempts)
        ]
        median_filled_rate = _median(setup_fill_rates)
        if median_filled_rate <= 0.0:
            median_filled_rate = _safe_float(record.get("filled_rate"), 0.02) or 0.02
        trade_metrics_for_strategy = trade_metrics_by_setup.get(strategy_key, {})
        lanes: list[dict[str, Any]] = []
        for item in sorted(
            setups_raw,
            key=lambda value: (
                -_safe_int(value.get("attempts"), 0),
                str(value.get("setup_key") or ""),
            ),
        ):
            if not isinstance(item, dict):
                continue
            context = {
                "setup_fingerprint": str(item.get("setup_fingerprint") or "").strip(),
                "flow_regime": str(item.get("flow_regime") or "").strip(),
                "microstructure_bucket": str(
                    item.get("microstructure_bucket") or ""
                ).strip(),
            }
            if not any(context.values()):
                continue
            setup_key = _setup_key(strategy_key, context)
            trade_metrics = trade_metrics_for_strategy.get(setup_key, {})
            attempts = _safe_int(item.get("attempts"), 0)
            fills = _safe_int(item.get("fills"), 0)
            filled_rate = _safe_float(item.get("filled_rate"), 0.0)
            attempt_share = _safe_float(item.get("attempt_share"), 0.0)
            fill_share = _safe_float(item.get("fill_share"), 0.0)
            share_gap = _safe_float(item.get("share_gap"), attempt_share - fill_share)
            hard_block_rate = _safe_float(item.get("hard_block_rate"), 0.0)
            promotion_gate = _promotion_gate(
                attempts=attempts,
                fills=fills,
                filled_rate=filled_rate,
                median_filled_rate=median_filled_rate,
                attempt_share=attempt_share,
                fill_share=fill_share,
                hard_block_rate=hard_block_rate,
                trade_metrics=trade_metrics,
                min_attempts=max(2, setup_min_attempts),
                max_units_boost=max_units_boost,
                max_prob_boost=max_prob_boost,
            )
            quarantine_gate = _quarantine_gate(
                attempts=attempts,
                fills=fills,
                filled_rate=filled_rate,
                median_filled_rate=median_filled_rate,
                share_gap=share_gap,
                hard_block_rate=hard_block_rate,
                trade_metrics=trade_metrics,
                min_attempts=max(2, setup_min_attempts),
                max_units_cut=max_units_cut,
                max_prob_boost=max_prob_boost,
            )
            if quarantine_gate["active"]:
                action = "trim_units"
                decision = quarantine_gate
                gate_action = "quarantine"
            elif promotion_gate["passed"]:
                action = "boost_participation"
                decision = promotion_gate
                gate_action = "promote"
            else:
                action = "hold"
                decision = {
                    "units_multiplier": 1.0,
                    "lot_multiplier": 1.0,
                    "probability_boost": 0.0,
                    "probability_offset": 0.0,
                    "max_probability_cut": 0.0,
                    "cadence_floor": 1.0,
                }
                gate_action = "hold"
            quality_score = _quality_score(
                filled_rate=filled_rate,
                median_filled_rate=median_filled_rate,
                attempt_share=attempt_share,
                fill_share=fill_share,
                realized_jpy=_safe_float(trade_metrics.get("realized_jpy"), 0.0),
                profit_factor=_safe_float(trade_metrics.get("profit_factor"), 0.0),
                win_rate=_safe_float(trade_metrics.get("win_rate"), 0.0),
            )
            lane = {
                "setup_key": str(item.get("setup_key") or ""),
                "match_dimension": str(
                    item.get("match_dimension") or "setup_fingerprint"
                ),
                "setup_fingerprint": context["setup_fingerprint"],
                "flow_regime": context["flow_regime"],
                "microstructure_bucket": context["microstructure_bucket"],
                "attempts": attempts,
                "preflights": attempts,
                "fills": fills,
                "filled": fills,
                "filled_rate": round(filled_rate, 6),
                "fill_rate": round(filled_rate, 6),
                "attempt_share": round(attempt_share, 6),
                "fill_share": round(fill_share, 6),
                "share_gap": round(share_gap, 6),
                "hard_block_rate": round(hard_block_rate, 6),
                "quality_score": quality_score,
                **trade_metrics,
                "gate_action": gate_action,
                "action": action,
                "units_multiplier": round(
                    _clamp(
                        _safe_float(decision.get("units_multiplier"), 1.0),
                        1.0 - max_units_cut,
                        1.0 + max_units_boost,
                    ),
                    4,
                ),
                "lot_multiplier": round(
                    _clamp(
                        _safe_float(decision.get("lot_multiplier"), 1.0),
                        1.0 - max_units_cut,
                        1.0 + max_units_boost,
                    ),
                    4,
                ),
                "probability_boost": round(
                    _clamp(
                        _safe_float(decision.get("probability_boost"), 0.0),
                        0.0,
                        max_prob_boost,
                    ),
                    4,
                ),
                "probability_offset": round(
                    _clamp(
                        _safe_float(decision.get("probability_offset"), 0.0),
                        -0.25,
                        max_prob_boost,
                    ),
                    4,
                ),
                "max_probability_cut": round(
                    _clamp(
                        _safe_float(decision.get("max_probability_cut"), 0.0), 0.0, 0.25
                    ),
                    4,
                ),
                "cadence_floor": round(
                    _clamp(_safe_float(decision.get("cadence_floor"), 1.0), 0.85, 1.18),
                    4,
                ),
                "promotion_gate": {
                    "passed": bool(promotion_gate.get("passed")),
                    "strength": round(
                        _safe_float(promotion_gate.get("strength"), 0.0), 4
                    ),
                    "reasons": list(promotion_gate.get("reasons") or []),
                },
                "quarantine_gate": {
                    "active": bool(quarantine_gate.get("active")),
                    "severity": round(
                        _safe_float(quarantine_gate.get("severity"), 0.0), 4
                    ),
                    "reasons": list(quarantine_gate.get("reasons") or []),
                },
            }
            lanes.append(lane)
            lane_count += 1
            action_counts[action] = action_counts.get(action, 0) + 1
        strategy_rows[strategy_key] = {
            "strategy_key": strategy_key,
            "pocket": str(record.get("pocket") or "").strip() or "unknown",
            "median_filled_rate": round(median_filled_rate, 6),
            "lanes": lanes,
        }

    return {
        "as_of": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "lookback_hours": _safe_float(entry_path_summary.get("lookback_hours"), 0.0),
        "policy": {
            "min_attempts": int(max(1, min_attempts)),
            "setup_min_attempts": int(max(1, setup_min_attempts)),
            "max_units_cut": round(float(max_units_cut), 4),
            "max_units_boost": round(float(max_units_boost), 4),
            "max_probability_boost": round(float(max_prob_boost), 4),
        },
        "summary": {
            "strategies": strategy_count,
            "lanes": lane_count,
            "action_counts": action_counts,
        },
        "strategies": strategy_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build lane scoreboard artifact")
    parser.add_argument(
        "--entry-path-summary", default="logs/entry_path_summary_latest.json"
    )
    parser.add_argument("--trades-db", default="logs/trades.db")
    parser.add_argument("--output", default="logs/lane_scoreboard_latest.json")
    parser.add_argument("--history", default="logs/lane_scoreboard_history.jsonl")
    parser.add_argument("--lookback-hours", type=float, default=6.0)
    parser.add_argument("--min-attempts", type=int, default=4)
    parser.add_argument("--setup-min-attempts", type=int, default=2)
    parser.add_argument("--max-units-cut", type=float, default=0.22)
    parser.add_argument("--max-units-boost", type=float, default=0.24)
    parser.add_argument("--max-probability-boost", type=float, default=0.10)
    args = parser.parse_args()

    summary = _read_json(Path(args.entry_path_summary).resolve())
    trade_metrics_by_setup = _load_trade_metrics_by_setup(
        Path(args.trades_db).resolve(),
        lookback_hours=float(args.lookback_hours),
    )
    payload = build_lane_scoreboard(
        summary,
        trade_metrics_by_setup=trade_metrics_by_setup,
        min_attempts=max(1, int(args.min_attempts)),
        setup_min_attempts=max(1, int(args.setup_min_attempts)),
        max_units_cut=_clamp(float(args.max_units_cut), 0.0, 0.5),
        max_units_boost=_clamp(float(args.max_units_boost), 0.0, 0.3),
        max_prob_boost=_clamp(float(args.max_probability_boost), 0.0, 0.15),
    )
    _write_json_atomic(Path(args.output).resolve(), payload)
    _append_jsonl(Path(args.history).resolve(), payload)
    print(
        f"[lane-scoreboard] wrote {Path(args.output).resolve()} "
        f"strategies={payload['summary']['strategies']} "
        f"lanes={payload['summary']['lanes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
