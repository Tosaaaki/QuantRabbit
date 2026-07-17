"""Causal daily-loss suppression overlay (operator goal: fewer negative days).

The overlay decides, at each trade's decision time, whether the day has
already realized enough loss to stop entering — using only exits that have
already happened by that decision clock.  Nothing is deleted after the
fact: skipped trades were never entered, and the skip condition is fully
computable live.  A predeclared small grid of stop levels is chosen on
TRAIN only; the chosen level is then replicated unchanged elsewhere.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

from quant_rabbit.adaptive_exact_s5_profit_engine import TradeOutcome

OVERLAY_POLICY = "CAUSAL_INTRADAY_REALIZED_LOSS_STOP_V1"
PREDECLARED_STOP_GRID_PIPS: tuple[float, ...] = (25.0, 50.0, 100.0)
TRAIN_NET_RETENTION_FLOOR = 0.7


def apply_causal_daily_stop(
    outcomes: Sequence[TradeOutcome], *, stop_pips: float
) -> tuple[tuple[TradeOutcome, ...], int]:
    """Skip entries once the day's already-exited trades reach the loss stop."""

    if not isinstance(stop_pips, (int, float)) or isinstance(stop_pips, bool):
        raise ValueError("stop_pips must be a number")
    if float(stop_pips) <= 0:
        raise ValueError("stop_pips must be positive")
    ordered = sorted(outcomes, key=lambda row: (row.decision_utc, row.pair, row.side))
    kept: list[TradeOutcome] = []
    skipped = 0
    for row in ordered:
        day = row.decision_utc.date()
        realized_so_far = sum(
            prior.realized_pips
            for prior in kept
            if prior.decision_utc.date() == day
            and prior.exit_utc <= row.decision_utc
        )
        if realized_so_far <= -float(stop_pips):
            skipped += 1
            continue
        kept.append(row)
    return tuple(kept), skipped


def daily_net_pips(outcomes: Sequence[TradeOutcome]) -> dict[date, float]:
    result: dict[date, float] = {}
    for row in outcomes:
        day = row.decision_utc.date()
        result[day] = result.get(day, 0.0) + row.realized_pips
    return {day: round(value, 9) for day, value in result.items()}


def negative_day_stats(daily: Mapping[date, float]) -> dict[str, Any]:
    values = list(daily.values())
    negatives = [value for value in values if value < 0.0]
    return {
        "active_days": len(values),
        "negative_days": len(negatives),
        "negative_day_rate": round(len(negatives) / len(values), 9) if values else 0.0,
        "worst_day_pips": round(min(values), 9) if values else 0.0,
        "mean_negative_day_pips": round(
            sum(negatives) / len(negatives), 9
        )
        if negatives
        else 0.0,
        "net_pips": round(sum(values), 9),
    }


def choose_stop_on_train(
    train_outcomes: Sequence[TradeOutcome],
) -> dict[str, Any]:
    """Pick one stop level from the predeclared grid using TRAIN only.

    Objective (declared before looking at results): minimize negative-day
    count, subject to retaining at least 70% of the baseline TRAIN net pips;
    ties break on shallower worst day, then higher net.  When no level
    satisfies the retention floor the overlay is rejected (baseline kept).
    """

    baseline_daily = daily_net_pips(train_outcomes)
    baseline = negative_day_stats(baseline_daily)
    candidates: list[dict[str, Any]] = []
    for stop in PREDECLARED_STOP_GRID_PIPS:
        kept, skipped = apply_causal_daily_stop(train_outcomes, stop_pips=stop)
        stats = negative_day_stats(daily_net_pips(kept))
        retention = (
            stats["net_pips"] / baseline["net_pips"]
            if baseline["net_pips"] > 0
            else float("-inf")
        )
        candidates.append(
            {
                "stop_pips": stop,
                "skipped_trades": skipped,
                "stats": stats,
                "net_retention_vs_baseline": round(retention, 9),
                "retention_floor_met": retention >= TRAIN_NET_RETENTION_FLOOR,
            }
        )
    eligible = [row for row in candidates if row["retention_floor_met"]]
    chosen = (
        min(
            eligible,
            key=lambda row: (
                row["stats"]["negative_days"],
                row["stats"]["worst_day_pips"] * -1.0,
                -row["stats"]["net_pips"],
            ),
        )
        if eligible
        else None
    )
    return {
        "overlay_policy": OVERLAY_POLICY,
        "predeclared_grid_pips": list(PREDECLARED_STOP_GRID_PIPS),
        "train_baseline": baseline,
        "candidates": candidates,
        "chosen_stop_pips": chosen["stop_pips"] if chosen else None,
        "overlay_rejected_no_retention": chosen is None,
        "selection_objective": (
            "MIN_NEGATIVE_DAYS_SUBJECT_TO_70PCT_NET_RETENTION_"
            "THEN_SHALLOWEST_WORST_DAY_THEN_NET_V1"
        ),
        "zero_negative_days_guaranteed": False,
    }
