from __future__ import annotations

from strategies.micro.trend_retest import MicroTrendRetest


def _flat_history(*, high: float, low: float, close: float) -> list[dict[str, float]]:
    candles: list[dict[str, float]] = []
    for _ in range(MicroTrendRetest._LOOKBACK):
        candles.append({"high": high, "low": low, "close": close})
    return candles


def _short_fac(
    *,
    prev_close: float,
    last_high: float,
    last_low: float,
    last_close: float,
    price: float,
    rsi: float,
    atr_pips: float = 4.0,
) -> dict[str, object]:
    return {
        "close": price,
        "ma10": 99.92,
        "ma20": 100.02,
        "adx": 26.0,
        "atr_pips": atr_pips,
        "spread_pips": 0.4,
        "rsi": rsi,
        "candles": _flat_history(high=100.18, low=100.00, close=100.08)
        + [
            {"high": 100.04, "low": min(prev_close, 99.95), "close": prev_close},
            {"high": last_high, "low": last_low, "close": last_close},
        ],
    }


def _long_fac(
    *,
    prev_close: float,
    last_high: float,
    last_low: float,
    last_close: float,
    price: float,
    rsi: float,
    atr_pips: float = 4.0,
) -> dict[str, object]:
    return {
        "close": price,
        "ma10": 100.08,
        "ma20": 99.98,
        "adx": 26.0,
        "atr_pips": atr_pips,
        "spread_pips": 0.4,
        "rsi": rsi,
        "candles": _flat_history(high=100.00, low=99.82, close=99.92)
        + [
            {"high": max(prev_close, 100.05), "low": 100.00, "close": prev_close},
            {"high": last_high, "low": last_low, "close": last_close},
        ],
    }


def test_short_retest_allows_when_rsi_still_reflects_pullback_state() -> None:
    signal = MicroTrendRetest.check(
        _short_fac(
            prev_close=99.99,
            last_high=100.002,
            last_low=99.986,
            last_close=99.998,
            price=99.994,
            rsi=53.0,
        )
    )

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"


def test_short_retest_rejects_when_indicators_already_show_resumed_breakdown() -> None:
    signal = MicroTrendRetest.check(
        _short_fac(
            prev_close=99.99,
            last_high=100.002,
            last_low=99.986,
            last_close=99.998,
            price=99.994,
            rsi=44.0,
        )
    )

    assert signal is None


def test_short_retest_rejects_strong_opposing_higher_tf_snapshot() -> None:
    fac = _short_fac(
        prev_close=99.99,
        last_high=100.002,
        last_low=99.986,
        last_close=99.998,
        price=99.994,
        rsi=53.0,
    )
    fac["trend_snapshot"] = {
        "tf": "H4",
        "direction": "long",
        "gap_pips": 33.692,
        "adx": 31.52,
    }

    signal = MicroTrendRetest.check(fac)

    assert signal is None


def test_long_retest_allows_when_rsi_still_reflects_pullback_state() -> None:
    signal = MicroTrendRetest.check(
        _long_fac(
            prev_close=100.01,
            last_high=100.014,
            last_low=99.998,
            last_close=100.002,
            price=100.006,
            rsi=47.0,
        )
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"


def test_long_retest_rejects_when_indicators_already_show_resumed_breakout() -> None:
    signal = MicroTrendRetest.check(
        _long_fac(
            prev_close=100.01,
            last_high=100.014,
            last_low=99.998,
            last_close=100.002,
            price=100.006,
            rsi=56.0,
        )
    )

    assert signal is None


def test_long_retest_rejects_strong_opposing_higher_tf_snapshot() -> None:
    fac = _long_fac(
        prev_close=100.01,
        last_high=100.014,
        last_low=99.998,
        last_close=100.002,
        price=100.006,
        rsi=47.0,
    )
    fac["trend_snapshot"] = {
        "tf": "H4",
        "direction": "short",
        "gap_pips": -24.0,
        "adx": 31.0,
    }

    signal = MicroTrendRetest.check(fac)

    assert signal is None


def test_long_retest_rejects_low_atr_when_close_sticks_to_retest_low() -> None:
    signal = MicroTrendRetest.check(
        _long_fac(
            prev_close=100.01,
            last_high=100.010,
            last_low=99.998,
            last_close=100.000,
            price=100.004,
            rsi=47.0,
            atr_pips=3.0,
        )
    )

    assert signal is None


def test_long_retest_allows_low_atr_when_close_recovers_off_retest_low() -> None:
    signal = MicroTrendRetest.check(
        _long_fac(
            prev_close=100.01,
            last_high=100.010,
            last_low=99.998,
            last_close=100.008,
            price=100.006,
            rsi=47.0,
            atr_pips=3.0,
        )
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"


def test_short_retest_rejects_low_atr_when_close_sticks_to_retest_high() -> None:
    signal = MicroTrendRetest.check(
        _short_fac(
            prev_close=99.99,
            last_high=100.002,
            last_low=99.990,
            last_close=100.000,
            price=99.996,
            rsi=53.0,
            atr_pips=3.0,
        )
    )

    assert signal is None
