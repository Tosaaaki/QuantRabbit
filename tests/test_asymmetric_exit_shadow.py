from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.asymmetric_exit_shadow import (
    AsymmetricExitError,
    resolve_structure_break_exit,
)
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle

UTC = timezone.utc
FILL_AT = datetime(2026, 7, 15, 9, 0, 5, tzinfo=UTC)
SHA = "a" * 64


def _candle(
    offset_seconds: int,
    *,
    bid_o: float,
    bid_h: float | None = None,
    bid_l: float | None = None,
    spread: float = 0.0002,
) -> S5BidAskCandle:
    high = bid_h if bid_h is not None else bid_o
    low = bid_l if bid_l is not None else bid_o
    close = bid_o
    return S5BidAskCandle(
        timestamp_utc=FILL_AT.replace(second=5) + timedelta(seconds=offset_seconds),
        bid_o=bid_o,
        bid_h=high,
        bid_l=low,
        bid_c=close,
        ask_o=bid_o + spread,
        ask_h=high + spread,
        ask_l=low + spread,
        ask_c=close + spread,
    )


def _resolve(candles, **overrides):
    values = {
        "side": "LONG",
        "fill_price": 1.1000,
        "fill_at_utc": FILL_AT,
        "structure_level": 1.0980,
        "structure_provenance_sha256": SHA,
        "time_boundary_utc": FILL_AT + timedelta(minutes=10),
        "candles": candles,
        "pip_factor": 10_000.0,
    }
    values.update(overrides)
    return resolve_structure_break_exit(**values)


def test_winner_runs_until_time_boundary_with_no_tp_cap() -> None:
    candles = [
        _candle(0, bid_o=1.1000),
        _candle(300, bid_o=1.1030),
        _candle(605, bid_o=1.1060),
    ]
    outcome = _resolve(candles)
    assert outcome["status"] == "RESOLVED"
    assert outcome["exit_reason"] == "EXECUTABLE_TIME_CLOSE"
    # +60 pips captured: far beyond any fixed TP multiple of the 20-pip risk.
    assert outcome["realized_pips"] == pytest.approx(60.0)
    assert outcome["take_profit_exists"] is False


def test_structure_break_and_gap_are_charged_executably() -> None:
    touched = _resolve(
        [_candle(0, bid_o=1.1000), _candle(300, bid_o=1.0990, bid_l=1.0979)]
    )
    assert touched["exit_reason"] == "STRUCTURE_BREAK"
    assert touched["realized_pips"] == pytest.approx(-20.0)

    gapped = _resolve(
        [_candle(0, bid_o=1.1000), _candle(300, bid_o=1.0970, bid_l=1.0968)]
    )
    assert gapped["exit_reason"] == "STRUCTURE_BREAK_GAP"
    assert gapped["realized_pips"] == pytest.approx(-30.0)


def test_fill_candle_touch_is_pessimistic_and_coverage_gap_stays_usable() -> None:
    ambiguous = _resolve([_candle(0, bid_o=1.1000, bid_l=1.0979)])
    assert ambiguous["exit_reason"] == "STRUCTURE_BREAK_AMBIGUOUS_FILL_S5"
    assert ambiguous["ambiguous_same_s5"] is True
    assert ambiguous["realized_pips"] == pytest.approx(-20.0)

    unresolved = _resolve([_candle(0, bid_o=1.1000)])
    assert unresolved["status"] == "UNRESOLVED_INSUFFICIENT_COVERAGE"
    assert unresolved["cohort_must_use_pessimistic_value"] is True
    assert unresolved["pessimistic_realized_pips"] == pytest.approx(-20.0)


def test_geometry_and_provenance_are_fail_closed() -> None:
    with pytest.raises(AsymmetricExitError, match="below the fill"):
        _resolve([_candle(0, bid_o=1.1)], structure_level=1.2)
    with pytest.raises(AsymmetricExitError, match="sha256"):
        _resolve([_candle(0, bid_o=1.1)], structure_provenance_sha256="short")
    with pytest.raises(AsymmetricExitError, match="after the fill"):
        _resolve(
            [_candle(0, bid_o=1.1)],
            time_boundary_utc=FILL_AT - timedelta(minutes=1),
        )
    with pytest.raises(AsymmetricExitError, match="chronological"):
        _resolve([_candle(300, bid_o=1.1), _candle(300, bid_o=1.1)])
