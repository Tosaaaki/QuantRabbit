from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_rabbit.cost_window_mask import evaluate_cost_window
from quant_rabbit.nav_normalization import nav_columns_for_metrics, pips_to_nav_fraction

UTC = timezone.utc


def test_nav_fraction_respects_pip_value_differences() -> None:
    # 10 pips on 10k EUR_USD with USD account and NAV 1,000,000 JPY-equivalent
    # in USD terms 10_000 USD NAV: 10 * 0.0001 * 10_000 = 10 USD -> 0.1%.
    usd_pair = pips_to_nav_fraction(
        pair="EUR_USD",
        realized_pips=10.0,
        units=10_000,
        quote_to_account_rate=1.0,
        nav_account_currency=10_000.0,
    )
    assert usd_pair == pytest.approx(0.001)

    # The same 10 pips on USD_JPY (pip size 0.01, quote JPY) with a JPY->USD
    # rate of 1/150 lands differently: 10 * 0.01 * 10_000 / 150 = 6.67 USD.
    jpy_pair = pips_to_nav_fraction(
        pair="USD_JPY",
        realized_pips=10.0,
        units=10_000,
        quote_to_account_rate=1.0 / 150.0,
        nav_account_currency=10_000.0,
    )
    assert jpy_pair == pytest.approx(6.6667 / 10_000.0, rel=1e-3)
    assert jpy_pair != usd_pair

    columns = nav_columns_for_metrics(
        pair="EUR_USD",
        net_pips=100.0,
        trade_count=20,
        units=10_000,
        quote_to_account_rate=1.0,
        nav_account_currency=10_000.0,
    )
    assert columns["net_nav_fraction"] == pytest.approx(0.01)
    assert columns["mean_nav_fraction_per_trade"] == pytest.approx(0.0005)

    with pytest.raises(ValueError, match="positive"):
        pips_to_nav_fraction(
            pair="EUR_USD",
            realized_pips=1.0,
            units=0.0,
            quote_to_account_rate=1.0,
            nav_account_currency=10_000.0,
        )


def test_cost_window_mask_blocks_rollover_and_validates_windows() -> None:
    rollover = evaluate_cost_window(datetime(2026, 7, 15, 21, 30, tzinfo=UTC))
    assert rollover.admitted is False
    assert rollover.payload()["derived_from_live_prices"] is False

    open_hours = evaluate_cost_window(datetime(2026, 7, 15, 9, 0, tzinfo=UTC))
    assert open_hours.admitted is True

    custom = evaluate_cost_window(
        datetime(2026, 7, 17, 20, 30, tzinfo=UTC),
        masked_windows=((20 * 60, 22 * 60),),
    )
    assert custom.admitted is False

    with pytest.raises(ValueError, match="inside one UTC day"):
        evaluate_cost_window(
            datetime(2026, 7, 15, 9, 0, tzinfo=UTC),
            masked_windows=((23 * 60, 25 * 60),),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_cost_window(datetime(2026, 7, 15, 9, 0))
