"""Replay quality gate utilities.

This module provides reusable functions to evaluate replay outputs with
walk-forward folds and deterministic quality checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from statistics import median
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GateThreshold:
    min_train_trades: int = 0
    min_test_trades: int = 0
    min_test_pf: float = 0.0
    min_test_win_rate: float = 0.0
    min_test_total_pips: float = -1.0e9
    max_test_drawdown_pips: float = 1.0e9
    min_pf_stability_ratio: float = 0.0


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out):
        return float(default)
    return out


def _parse_iso8601(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _drawdown_from_pips(pips_series: Sequence[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pips_series:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_trade_metrics(
    trades: Sequence[Mapping[str, Any]],
    *,
    exclude_reason: str | None = None,
) -> dict[str, float]:
    rows: list[tuple[float, int, float]] = []

    for idx, trade in enumerate(trades):
        if exclude_reason and str(trade.get("reason") or "") == exclude_reason:
            continue
        pnl = _safe_float(trade.get("pnl_pips"), default=math.nan)
        if math.isnan(pnl):
            continue
        ts = _parse_iso8601(trade.get("exit_time")) or _parse_iso8601(trade.get("entry_time"))
        ts_key = ts.timestamp() if ts is not None else 0.0
        rows.append((ts_key, idx, pnl))

    rows.sort(key=lambda item: (item[0], item[1]))
    pips = [item[2] for item in rows]
    n = len(pips)
    if n == 0:
        return {
            "trade_count": 0.0,
            "total_pips": 0.0,
            "avg_pips": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pips": 0.0,
        }

    wins = [x for x in pips if x > 0.0]
    losses = [x for x in pips if x < 0.0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0.0:
        pf = gross_win / gross_loss
    elif gross_win > 0.0:
        pf = float("inf")
    else:
        pf = 0.0

    return {
        "trade_count": float(n),
        "total_pips": float(sum(pips)),
        "avg_pips": float(sum(pips) / n),
        "win_rate": float(len(wins) / n),
        "profit_factor": float(pf),
        "max_drawdown_pips": float(_drawdown_from_pips(pips)),
    }


def build_walk_forward_folds(
    ordered_items: Sequence[str],
    *,
    train_files: int,
    test_files: int,
    step_files: int,
) -> list[dict[str, list[str]]]:
    train_n = max(1, int(train_files))
    test_n = max(1, int(test_files))
    step_n = max(1, int(step_files))

    total = len(ordered_items)
    if total < train_n + test_n:
        return []

    folds: list[dict[str, list[str]]] = []
    cursor = train_n
    while cursor + test_n <= total:
        train_part = list(ordered_items[cursor - train_n : cursor])
        test_part = list(ordered_items[cursor : cursor + test_n])
        folds.append({"train": train_part, "test": test_part})
        cursor += step_n
    return folds


def _check_ge(name: str, actual: float, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "op": ">=",
        "actual": float(actual),
        "threshold": float(threshold),
        "ok": bool(actual >= threshold),
    }


def _check_le(name: str, actual: float, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "op": "<=",
        "actual": float(actual),
        "threshold": float(threshold),
        "ok": bool(actual <= threshold),
    }


def _pf_stability_ratio(train_pf: float, test_pf: float) -> float:
    if train_pf <= 0.0:
        return float("inf") if test_pf > 0.0 else 0.0
    if math.isinf(train_pf):
        return 1.0 if math.isinf(test_pf) else 0.0
    return test_pf / train_pf


def evaluate_fold_gate(
    train_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float],
    threshold: GateThreshold,
) -> dict[str, Any]:
    train_pf = _safe_float(train_metrics.get("profit_factor"), default=0.0)
    test_pf = _safe_float(test_metrics.get("profit_factor"), default=0.0)
    pf_stability_ratio = _pf_stability_ratio(train_pf, test_pf)

    checks = [
        _check_ge(
            "train_trade_count",
            _safe_float(train_metrics.get("trade_count"), default=0.0),
            float(max(0, threshold.min_train_trades)),
        ),
        _check_ge(
            "test_trade_count",
            _safe_float(test_metrics.get("trade_count"), default=0.0),
            float(max(0, threshold.min_test_trades)),
        ),
        _check_ge("test_profit_factor", test_pf, threshold.min_test_pf),
        _check_ge(
            "test_win_rate",
            _safe_float(test_metrics.get("win_rate"), default=0.0),
            threshold.min_test_win_rate,
        ),
        _check_ge(
            "test_total_pips",
            _safe_float(test_metrics.get("total_pips"), default=0.0),
            threshold.min_test_total_pips,
        ),
        _check_le(
            "test_max_drawdown_pips",
            _safe_float(test_metrics.get("max_drawdown_pips"), default=0.0),
            threshold.max_test_drawdown_pips,
        ),
        _check_ge("pf_stability_ratio", pf_stability_ratio, threshold.min_pf_stability_ratio),
    ]
    passed = all(bool(item.get("ok")) for item in checks)

    failed = [item["name"] for item in checks if not item["ok"]]
    return {
        "passed": passed,
        "checks": checks,
        "failed_checks": failed,
        "pf_stability_ratio": pf_stability_ratio,
        "threshold": asdict(threshold),
    }


def summarize_worker_folds(
    fold_results: Sequence[Mapping[str, Any]],
    *,
    min_fold_pass_rate: float,
) -> dict[str, Any]:
    evaluated = list(fold_results)
    total = len(evaluated)
    passed = sum(1 for item in evaluated if bool(item.get("gate", {}).get("passed")))
    pass_rate = (passed / total) if total > 0 else 0.0

    test_pf_values = [
        _safe_float(item.get("test_metrics", {}).get("profit_factor"), default=0.0)
        for item in evaluated
    ]
    test_wr_values = [
        _safe_float(item.get("test_metrics", {}).get("win_rate"), default=0.0)
        for item in evaluated
    ]
    test_dd_values = [
        _safe_float(item.get("test_metrics", {}).get("max_drawdown_pips"), default=0.0)
        for item in evaluated
    ]

    return {
        "folds": total,
        "passed_folds": passed,
        "pass_rate": float(pass_rate),
        "required_pass_rate": float(min_fold_pass_rate),
        "status": "pass" if total > 0 and pass_rate >= min_fold_pass_rate else "fail",
        "median_test_pf": float(median(test_pf_values)) if test_pf_values else 0.0,
        "median_test_win_rate": float(median(test_wr_values)) if test_wr_values else 0.0,
        "median_test_max_drawdown_pips": float(median(test_dd_values)) if test_dd_values else 0.0,
    }
