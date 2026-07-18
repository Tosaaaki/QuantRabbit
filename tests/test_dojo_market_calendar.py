from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.dojo_market_calendar import (
    expected_oanda_fx_slots,
    oanda_fx_candle_open_is_expected,
)


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_m1_and_m5_aligned_reopen_buckets_are_explicit() -> None:
    m1 = timedelta(minutes=1)
    assert oanda_fx_candle_open_is_expected(utc("2026-07-13T20:59:00Z"), step=m1)
    assert not oanda_fx_candle_open_is_expected(
        utc("2026-07-13T21:00:00Z"), step=m1
    )
    assert not oanda_fx_candle_open_is_expected(
        utc("2026-07-13T21:03:00Z"), step=m1
    )
    assert oanda_fx_candle_open_is_expected(utc("2026-07-13T21:04:00Z"), step=m1)
    assert oanda_fx_candle_open_is_expected(utc("2026-07-13T21:05:00Z"), step=m1)

    m5 = timedelta(minutes=5)
    assert oanda_fx_candle_open_is_expected(utc("2026-07-13T20:55:00Z"), step=m5)
    assert oanda_fx_candle_open_is_expected(utc("2026-07-13T21:00:00Z"), step=m5)
    assert oanda_fx_candle_open_is_expected(utc("2026-07-13T21:05:00Z"), step=m5)


def test_boundaries_follow_new_york_dst_and_friday_close() -> None:
    m1 = timedelta(minutes=1)
    assert oanda_fx_candle_open_is_expected(utc("2026-01-12T21:59:00Z"), step=m1)
    assert not oanda_fx_candle_open_is_expected(
        utc("2026-01-12T22:00:00Z"), step=m1
    )
    assert oanda_fx_candle_open_is_expected(utc("2026-01-12T22:04:00Z"), step=m1)
    assert oanda_fx_candle_open_is_expected(utc("2026-07-17T20:59:00Z"), step=m1)
    assert not oanda_fx_candle_open_is_expected(
        utc("2026-07-17T21:00:00Z"), step=m1
    )


def test_slot_window_must_use_absolute_oanda_grid() -> None:
    with pytest.raises(ValueError, match="absolute candle grid"):
        expected_oanda_fx_slots(
            utc("2026-07-13T00:01:00Z"),
            utc("2026-07-13T01:01:00Z"),
            step=timedelta(minutes=5),
        )
