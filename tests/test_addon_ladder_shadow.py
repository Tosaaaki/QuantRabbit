from __future__ import annotations

from array import array
from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.addon_ladder_shadow import resolve_addon_ladder
from quant_rabbit.adaptive_exact_s5_profit_engine import ExactS5Series, TradeOutcome

UTC = timezone.utc
ENTRY = datetime(2026, 5, 12, 4, 0, 5, tzinfo=UTC)
EXIT = ENTRY + timedelta(hours=1)
SPREAD = 0.0002


def _series(mids: list[float]) -> ExactS5Series:
    epochs = array("q")
    bids = array("d")
    asks = array("d")
    for index, mid in enumerate(mids):
        epochs.append(int(ENTRY.timestamp()) + 5 * (index + 1))
        bids.append(mid - SPREAD / 2)
        asks.append(mid + SPREAD / 2)
    return ExactS5Series(
        points=(), minute_epochs=(), s5_epochs=epochs, bid_opens=bids, ask_opens=asks
    )


def _outcome(*, entry_ask: float, exit_bid: float) -> TradeOutcome:
    return TradeOutcome(
        pair="EUR_USD",
        side="LONG",
        decision_utc=ENTRY,
        entry_utc=ENTRY,
        exit_utc=EXIT,
        score=0.0,
        raw_return_pips=0.0,
        entry_bid=entry_ask - SPREAD,
        entry_ask=entry_ask,
        exit_bid=exit_bid,
        exit_ask=exit_bid + SPREAD,
        gross_mid_pips=0.0,
        round_trip_spread_pips=2.0,
        realized_pips=(exit_bid - entry_ask) * 10_000,
        entry_delay_seconds=0,
        exit_delay_seconds=0,
    )


def test_pyramid_adds_on_favorable_s5_opens_and_blends() -> None:
    outcome = _outcome(entry_ask=1.1000, exit_bid=1.1060)
    series = _series([1.1000, 1.1021, 1.1041, 1.1055])

    result = resolve_addon_ladder(
        series, outcome, mode="PYRAMID", step_pips=20.0, max_adds=2, pip_factor=10_000.0
    )

    assert result["units_filled"] == 3
    assert result["peak_exposure_multiple"] == 3
    # Later adds enter at worse prices, so per-unit blended < base realized.
    assert result["blended_pips_per_unit"] < result["base_realized_pips"]
    assert result["blended_total_pips"] > result["base_realized_pips"]


def test_nanpin_is_hard_capped_and_deepens_loss_multiples() -> None:
    outcome = _outcome(entry_ask=1.1000, exit_bid=1.0950)
    series = _series([1.0990, 1.0979, 1.0959, 1.0949, 1.0940])

    result = resolve_addon_ladder(
        series, outcome, mode="NANPIN", step_pips=20.0, max_adds=2, pip_factor=10_000.0
    )

    assert result["units_filled"] == 3
    # Averaging down improves per-unit pips but multiplies total exposure loss.
    assert result["blended_pips_per_unit"] > result["base_realized_pips"]
    assert result["blended_total_pips"] < result["base_realized_pips"]

    with pytest.raises(ValueError, match="hard cap"):
        resolve_addon_ladder(
            series, outcome, mode="NANPIN", step_pips=20.0, max_adds=9,
            pip_factor=10_000.0,
        )
