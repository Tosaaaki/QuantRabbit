from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_rabbit.close_distance_gate import (
    evaluate_close_distance_gate,
    max_admissible_hold_minutes,
)
from quant_rabbit.currency_exposure_guard import (
    evaluate_currency_exposure,
    net_currency_exposure,
)

UTC = timezone.utc
# Mid-week London morning: market open, days until the Friday close.
MIDWEEK = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)
# Friday 20:00 UTC: the NY close is near.
FRIDAY_LATE = datetime(2026, 7, 17, 20, 0, tzinfo=UTC)
# Saturday: FX closed.
SATURDAY = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def test_gate_admits_midweek_and_refuses_close_crossing_holds() -> None:
    admitted = evaluate_close_distance_gate(MIDWEEK, hold_minutes=720)
    assert admitted.admitted is True
    assert admitted.payload()["uses_post_entry_information"] is False

    crossing = evaluate_close_distance_gate(FRIDAY_LATE, hold_minutes=720)
    assert crossing.admitted is False
    assert crossing.reason == "HOLD_WOULD_CROSS_NEXT_FX_CLOSE"

    closed = evaluate_close_distance_gate(SATURDAY, hold_minutes=30)
    assert closed.admitted is False
    assert closed.reason == "FX_MARKET_CLOSED"


def test_max_admissible_hold_shrinks_near_the_close() -> None:
    assert max_admissible_hold_minutes(SATURDAY) == 0
    late_cap = max_admissible_hold_minutes(FRIDAY_LATE)
    assert 0 < late_cap < 720
    # The reported cap must itself be admissible.
    assert evaluate_close_distance_gate(
        FRIDAY_LATE, hold_minutes=late_cap
    ).admitted is True

    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_close_distance_gate(
            MIDWEEK.replace(tzinfo=None), hold_minutes=60
        )


def test_net_exposure_cancels_and_caps_theme_stacking() -> None:
    positions = [
        {"pair": "EUR_USD", "side": "LONG", "nav_exposure_fraction": 0.2},
        {"pair": "USD_JPY", "side": "SHORT", "nav_exposure_fraction": 0.2},
    ]
    exposure = net_currency_exposure(positions)
    # Both positions are short USD: the theme stacks to -0.4.
    assert exposure["USD"] == pytest.approx(-0.4)
    assert exposure["EUR"] == pytest.approx(0.2)
    assert exposure["JPY"] == pytest.approx(0.2)

    candidate = {"pair": "GBP_USD", "side": "LONG", "nav_exposure_fraction": 0.2}
    verdict = evaluate_currency_exposure(
        positions, candidate, currency_cap_fraction=0.5
    )
    assert verdict.admitted is False
    assert verdict.breached_currencies == ("USD",)

    hedged = {"pair": "USD_CHF", "side": "LONG", "nav_exposure_fraction": 0.2}
    assert evaluate_currency_exposure(
        positions, hedged, currency_cap_fraction=0.5
    ).admitted is True


def test_exposure_guard_rejects_malformed_positions() -> None:
    with pytest.raises(ValueError, match="pair identity"):
        net_currency_exposure([{ "pair": "EURUSD", "side": "LONG", "nav_exposure_fraction": 0.1 }])
    with pytest.raises(ValueError, match="side"):
        net_currency_exposure([{ "pair": "EUR_USD", "side": "BUY", "nav_exposure_fraction": 0.1 }])
    with pytest.raises(ValueError, match="positive"):
        net_currency_exposure([{ "pair": "EUR_USD", "side": "LONG", "nav_exposure_fraction": 0.0 }])
