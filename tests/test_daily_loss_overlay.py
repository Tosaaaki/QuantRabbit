from __future__ import annotations

from datetime import datetime, timedelta, timezone

from quant_rabbit.adaptive_exact_s5_profit_engine import TradeOutcome
from quant_rabbit.daily_loss_overlay import (
    apply_causal_daily_stop,
    choose_stop_on_train,
    daily_net_pips,
    negative_day_stats,
)

UTC = timezone.utc
DAY = datetime(2026, 5, 12, tzinfo=UTC)


def _trade(
    decision: datetime, *, pips: float, hold_minutes: int = 60, pair: str = "EUR_USD"
) -> TradeOutcome:
    return TradeOutcome(
        pair=pair,
        side="LONG",
        decision_utc=decision,
        entry_utc=decision,
        exit_utc=decision + timedelta(minutes=hold_minutes),
        score=0.0,
        raw_return_pips=0.0,
        entry_bid=1.0,
        entry_ask=1.0,
        exit_bid=1.0,
        exit_ask=1.0,
        gross_mid_pips=pips,
        round_trip_spread_pips=0.0,
        realized_pips=pips,
        entry_delay_seconds=0,
        exit_delay_seconds=0,
    )


def test_stop_skips_only_after_realized_loss_is_knowable() -> None:
    trades = [
        _trade(DAY.replace(hour=0), pips=-30.0, hold_minutes=60),
        # Decision at 02:00 sees the 01:00 exit of the -30 trade.
        _trade(DAY.replace(hour=2), pips=50.0),
        # Decision at 00:30 does NOT see it (exit 01:00 is later).
        _trade(DAY.replace(hour=0, minute=30), pips=-5.0),
    ]

    kept, skipped = apply_causal_daily_stop(trades, stop_pips=25.0)

    kept_hours = sorted(row.decision_utc.hour for row in kept)
    assert kept_hours == [0, 0]
    assert skipped == 1  # only the 02:00 decision is stopped

    # HALF_SIZE keeps the 02:00 winner at half participation instead of
    # forfeiting it — the opportunity-cost-preserving variant.
    halved, affected = apply_causal_daily_stop(
        trades, stop_pips=25.0, mode="HALF_SIZE"
    )
    assert affected == 1
    late = next(row for row in halved if row.decision_utc.hour == 2)
    assert late.realized_pips == 25.0


def test_selection_reduces_negative_days_within_retention_floor() -> None:
    trades = []
    # Day 1: early knowable loss then a big later loss the stop avoids.
    trades.append(_trade(DAY, pips=-40.0, hold_minutes=30))
    trades.append(_trade(DAY.replace(hour=6), pips=-80.0))
    # Days 2-4: steady winners that fund retention.
    for day_offset in (1, 2, 3):
        base = DAY + timedelta(days=day_offset)
        trades.append(_trade(base, pips=60.0))
        trades.append(_trade(base.replace(hour=6), pips=40.0))

    selection = choose_stop_on_train(trades)

    assert selection["chosen_stop_pips"] == 25.0
    chosen = next(
        row
        for row in selection["candidates"]
        if row["stop_pips"] == selection["chosen_stop_pips"]
    )
    assert chosen["stats"]["worst_day_pips"] == -40.0  # -80 was never entered
    assert selection["train_baseline"]["worst_day_pips"] == -120.0
    assert selection["zero_negative_days_guaranteed"] is False

    stats = negative_day_stats(daily_net_pips(trades))
    assert stats["negative_days"] == 1
    assert stats["active_days"] == 4
