"""Causal daily-loss suppression overlay (operator goal: fewer negative days).

The overlay decides, at each trade's decision time, whether the day has
already realized enough loss to stop entering — using only exits that have
already happened by that decision clock.  Nothing is deleted after the
fact: skipped trades were never entered, and the skip condition is fully
computable live.  A predeclared small grid of stop levels is chosen on
TRAIN only; the chosen level is then replicated unchanged elsewhere.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any, Mapping, Sequence

from quant_rabbit.adaptive_exact_s5_profit_engine import TradeOutcome

OVERLAY_POLICY = "CAUSAL_INTRADAY_REALIZED_LOSS_STOP_V1"
PREDECLARED_STOP_GRID_PIPS: tuple[float, ...] = (25.0, 50.0, 100.0)
TRAIN_NET_RETENTION_FLOOR = 0.7


def apply_causal_daily_stop(
    outcomes: Sequence[TradeOutcome],
    *,
    stop_pips: float,
    mode: str = "SKIP",
) -> tuple[tuple[TradeOutcome, ...], int]:
    """Throttle entries once the day's already-exited trades reach the stop.

    ``SKIP`` stops entering for the rest of the day; ``HALF_SIZE`` keeps
    participating at half size (half the realized pips), so later winners
    are not forfeited — the operator's opportunity-cost objection made
    executable.  Both read only exits known at each decision time.
    """

    if not isinstance(stop_pips, (int, float)) or isinstance(stop_pips, bool):
        raise ValueError("stop_pips must be a number")
    if float(stop_pips) <= 0:
        raise ValueError("stop_pips must be positive")
    if mode not in {"SKIP", "HALF_SIZE"}:
        raise ValueError("mode must be SKIP or HALF_SIZE")
    ordered = sorted(outcomes, key=lambda row: (row.decision_utc, row.pair, row.side))
    kept: list[TradeOutcome] = []
    affected = 0
    for row in ordered:
        day = row.decision_utc.date()
        realized_so_far = sum(
            prior.realized_pips
            for prior in kept
            if prior.decision_utc.date() == day
            and prior.exit_utc <= row.decision_utc
        )
        if realized_so_far <= -float(stop_pips):
            affected += 1
            if mode == "SKIP":
                continue
            kept.append(
                replace(
                    row,
                    realized_pips=row.realized_pips * 0.5,
                    gross_mid_pips=row.gross_mid_pips * 0.5,
                    round_trip_spread_pips=row.round_trip_spread_pips * 0.5,
                )
            )
            continue
        kept.append(row)
    return tuple(kept), affected


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


def apply_causal_day_skip(
    outcomes: Sequence[TradeOutcome], *, prev_day_known_loss_pips: float
) -> tuple[tuple[TradeOutcome, ...], int]:
    """Skip a whole day when yesterday's KNOWN realized loss is deep enough.

    At day D 00:00 UTC only trades that have already exited are knowable, so
    the filter reads exactly that: the realized pips of previous-day
    decisions whose exits landed before D 00:00.  Long holds mean this is a
    partial picture — that partiality is the honest live condition.
    """

    if (
        not isinstance(prev_day_known_loss_pips, (int, float))
        or isinstance(prev_day_known_loss_pips, bool)
        or float(prev_day_known_loss_pips) < 0
    ):
        raise ValueError("prev_day_known_loss_pips must be non-negative")
    ordered = sorted(outcomes, key=lambda row: (row.decision_utc, row.pair, row.side))
    kept: list[TradeOutcome] = []
    skipped = 0
    skipped_days: set[date] = set()
    for row in ordered:
        day = row.decision_utc.date()
        if day in skipped_days:
            skipped += 1
            continue
        day_start = row.decision_utc.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # The reference day is the most recent day actually traded after
        # filtering (a skipped day produces no trades live), and only its
        # exits landed before today count — the honest live knowledge.
        last_traded = max(
            (prior.decision_utc.date() for prior in kept if prior.decision_utc.date() < day),
            default=None,
        )
        known_prev = (
            sum(
                prior.realized_pips
                for prior in kept
                if prior.decision_utc.date() == last_traded
                and prior.exit_utc <= day_start
            )
            if last_traded is not None
            else 0.0
        )
        if known_prev <= -float(prev_day_known_loss_pips):
            skipped_days.add(day)
            skipped += 1
            continue
        kept.append(row)
    return tuple(kept), skipped


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
