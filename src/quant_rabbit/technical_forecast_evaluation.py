"""Leakage-resistant evaluation helpers for causal technical forecasts."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence


TECHNICAL_FORECAST_EVALUATION_CONTRACT = (
    "QR_CAUSAL_TECHNICAL_FORECAST_EVALUATION_V1"
)


def select_non_overlapping_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon_min: float,
    minimum_absolute_prediction_pips: float,
) -> list[dict[str, Any]]:
    """Take the earliest qualifying forecast after each pair is flat again."""

    horizon = _positive(horizon_min, name="horizon_min")
    threshold = _non_negative(
        minimum_absolute_prediction_pips,
        name="minimum_absolute_prediction_pips",
    )
    accepted_until: dict[str, datetime] = {}
    selected: list[dict[str, Any]] = []
    normalized = []
    for source_index, raw in enumerate(rows):
        pair = str(raw.get("pair") or "").strip().upper()
        timestamp = _timestamp(raw.get("timestamp_utc"))
        prediction = _finite(raw.get("predicted_pips"))
        long_pips = _finite(raw.get("long_pips"))
        short_pips = _finite(raw.get("short_pips"))
        if (
            not pair
            or timestamp is None
            or prediction is None
            or long_pips is None
            or short_pips is None
        ):
            continue
        normalized.append(
            (
                timestamp,
                pair,
                source_index,
                prediction,
                long_pips,
                short_pips,
                raw,
            )
        )
    for timestamp, pair, _index, prediction, long_pips, short_pips, raw in sorted(
        normalized
    ):
        if abs(prediction) < threshold:
            continue
        if timestamp < accepted_until.get(pair, timestamp):
            continue
        direction = "UP" if prediction >= 0.0 else "DOWN"
        selected.append(
            {
                **dict(raw),
                "pair": pair,
                "timestamp_utc": timestamp.isoformat(),
                "predicted_pips": prediction,
                "predicted_direction": direction,
                "executed_pips": long_pips if direction == "UP" else short_pips,
                "evaluation_contract": TECHNICAL_FORECAST_EVALUATION_CONTRACT,
            }
        )
        accepted_until[pair] = timestamp + timedelta(minutes=horizon)
    return selected


def directional_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize exact executable returns from already-selected forecasts."""

    values = [
        value
        for row in rows
        if (value := _finite(row.get("executed_pips"))) is not None
    ]
    by_day: dict[str, list[float]] = defaultdict(list)
    by_pair: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get("executed_pips"))
        if value is None:
            continue
        timestamp = _timestamp(row.get("timestamp_utc"))
        pair = str(row.get("pair") or "").upper()
        if timestamp is not None:
            by_day[timestamp.date().isoformat()].append(value)
        if pair:
            by_pair[pair].append(value)
    daily = [sum(items) for items in by_day.values()]
    pair_metrics = {
        pair: _basic_metrics(items)
        for pair, items in sorted(by_pair.items())
    }
    return {
        **_basic_metrics(values),
        "active_days": len(daily),
        "positive_day_rate": round(
            sum(value > 0.0 for value in daily) / len(daily), 6
        )
        if daily
        else 0.0,
        "one_sided_95_daily_lower_pips": _one_sided_lower(daily),
        "by_pair": pair_metrics,
    }


def choose_validation_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon_min: float,
    thresholds_pips: Sequence[float],
    minimum_trades: int,
    minimum_active_days: int,
) -> dict[str, Any]:
    """Choose once on validation data; callers must lock it for holdout."""

    candidates: list[dict[str, Any]] = []
    for raw_threshold in thresholds_pips:
        threshold = _non_negative(
            raw_threshold,
            name="minimum_absolute_prediction_pips",
        )
        selected = select_non_overlapping_predictions(
            rows,
            horizon_min=horizon_min,
            minimum_absolute_prediction_pips=threshold,
        )
        metrics = directional_metrics(selected)
        enough = (
            metrics["trades"] >= int(minimum_trades)
            and metrics["active_days"] >= int(minimum_active_days)
        )
        candidates.append(
            {
                "threshold_pips": threshold,
                "minimum_evidence_met": enough,
                "metrics": metrics,
            }
        )
    eligible = [row for row in candidates if row["minimum_evidence_met"]]
    ranked = eligible or candidates
    ranked.sort(
        key=lambda row: (
            _rank_value(row["metrics"].get("one_sided_95_daily_lower_pips")),
            _rank_value(row["metrics"].get("one_sided_95_mean_lower_pips")),
            _rank_value(row["metrics"].get("mean_pips")),
            int(row["metrics"].get("trades") or 0),
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked else None
    return {
        "selected": selected,
        "candidates": candidates,
    }


def _basic_metrics(values: Sequence[float]) -> dict[str, Any]:
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    if gross_loss == 0.0:
        profit_factor: float | None = math.inf if gross_profit > 0.0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss
    return {
        "trades": len(values),
        "mean_pips": round(statistics.mean(values), 6) if values else None,
        "net_pips": round(sum(values), 6),
        "win_rate": round(len(wins) / len(values), 6) if values else 0.0,
        "profit_factor": round(profit_factor, 6)
        if profit_factor is not None and math.isfinite(profit_factor)
        else profit_factor,
        "one_sided_95_mean_lower_pips": _one_sided_lower(values),
    }


def _one_sided_lower(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    lower = statistics.mean(values) - 1.645 * statistics.stdev(values) / math.sqrt(
        len(values)
    )
    return round(lower, 6)


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _positive(value: Any, *, name: str) -> float:
    result = _finite(value)
    if result is None or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _non_negative(value: Any, *, name: str) -> float:
    result = _finite(value)
    if result is None or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _rank_value(value: Any) -> float:
    result = _finite(value)
    return result if result is not None else -math.inf
