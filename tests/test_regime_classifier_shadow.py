from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.regime_classifier_shadow import (
    RegimeClassifierError,
    _canonical_sha,
    classify_regime,
)

UTC = timezone.utc
START = datetime(2026, 5, 12, tzinfo=UTC)
AS_OF = START + timedelta(minutes=200)


def _candles(closes: list[float]) -> list[dict]:
    return [
        {"time": START + timedelta(minutes=i), "close": c}
        for i, c in enumerate(closes)
    ]


def test_trending_series_classifies_trend() -> None:
    closes = [1.1000 + i * 0.0002 for i in range(140)]  # steady uptrend
    result = classify_regime(_candles(closes), as_of_utc=AS_OF)

    assert result["regime"] == "TREND"
    assert result["components"]["efficiency_ratio"] >= 0.35
    body = {k: v for k, v in result.items() if k != "classification_sha256"}
    assert result["classification_sha256"] == _canonical_sha(body)


def test_choppy_wide_series_classifies_range() -> None:
    # Tight history then a WIDE choppy recent window: non-trending and
    # non-compressed (high BB-width percentile) -> RANGE.
    tight = [1.1000 + (0.0004 if i % 2 else -0.0004) for i in range(120)]
    wide = [1.1000 + (0.0030 if i % 2 else -0.0030) for i in range(20)]
    result = classify_regime(_candles(tight + wide), as_of_utc=AS_OF)

    assert result["regime"] == "RANGE"
    assert result["components"]["efficiency_ratio"] < 0.35
    assert result["components"]["bb_width_percentile"] >= 0.20


def test_recent_compression_relative_to_history_classifies_squeeze() -> None:
    # Wide choppy history, then a compressed (low BB-width percentile),
    # non-trending recent window -> SQUEEZE by relative compression.
    wide = [1.1000 + (0.0030 if i % 2 else -0.0030) for i in range(120)]
    tight = [1.1000 + (0.00005 if i % 2 else -0.00005) for i in range(20)]
    result = classify_regime(_candles(wide + tight), as_of_utc=AS_OF)

    assert result["regime"] == "SQUEEZE"
    assert result["components"]["bb_width_percentile"] < 0.20
    assert result["components"]["efficiency_ratio"] < 0.35


def test_event_flag_overrides_price_classification() -> None:
    closes = [1.1000 + i * 0.0002 for i in range(140)]
    result = classify_regime(
        _candles(closes), as_of_utc=AS_OF, high_impact_event_active=True
    )
    assert result["regime"] == "EVENT"
    assert result["confidence"] == 1.0


def test_high_vol_bucket_detects_volatility_expansion() -> None:
    # Calm history, then a volatile recent window -> HIGH vol percentile.
    calm = [1.1000 + 0.00001 * (1 if i % 2 else -1) for i in range(120)]
    volatile = [1.1000 + 0.0025 * (1 if i % 2 else -1) for i in range(20)]
    result = classify_regime(_candles(calm + volatile), as_of_utc=AS_OF)
    assert result["vol_state"] == "HIGH"


def test_frozen_feed_is_unclear_not_high_confidence_squeeze() -> None:
    # A dead/constant feed must not be a tradeable regime.
    flat = [1.1000] * 140
    result = classify_regime(_candles(flat), as_of_utc=AS_OF)
    assert result["regime"] == "UNCLEAR"
    assert result["confidence"] == 0.0
    assert result["degenerate_constant_window"] is True


def test_percentile_is_independent_of_caller_history_length() -> None:
    # A longer prefix must not change the classification: the rolling window
    # is bounded internally, so the same recent bars yield the same result.
    import random

    rng = random.Random(7)
    full_closes = [1.1000 + rng.uniform(-0.003, 0.003) for _ in range(400)]
    full = _candles(full_closes)
    as_of = full[-1]["time"] + timedelta(minutes=2)
    short = full[-145:]  # same absolute bars, shorter prefix
    r_full = classify_regime(full, as_of_utc=as_of)
    r_short = classify_regime(short, as_of_utc=as_of)
    assert r_full["regime"] == r_short["regime"]
    assert r_full["vol_state"] == r_short["vol_state"]
    assert r_full["components"] == r_short["components"]


def test_open_keyed_candle_must_be_closed_before_decision() -> None:
    closes = [1.1000 + i * 0.0002 for i in range(140)]
    candles = _candles(closes)
    # The last candle opens 30s before the decision: with a 60s period it has
    # not closed yet -> non-causal, must fail closed.
    near = AS_OF - timedelta(seconds=30)
    candles[-1]["time"] = near
    with pytest.raises(RegimeClassifierError, match="close at or before"):
        classify_regime(candles, as_of_utc=AS_OF, candle_period_seconds=60)


def test_causality_and_shape_fail_closed() -> None:
    closes = [1.1000 + i * 0.0002 for i in range(140)]
    # A candle at/after the decision clock is non-causal.
    future = _candles(closes)
    future[-1]["time"] = AS_OF
    with pytest.raises(RegimeClassifierError, match="close at or before"):
        classify_regime(future, as_of_utc=AS_OF)

    with pytest.raises(RegimeClassifierError, match="insufficient"):
        classify_regime(_candles(closes[:50]), as_of_utc=AS_OF)

    dupe = _candles(closes)
    dupe[10]["time"] = dupe[9]["time"]
    with pytest.raises(RegimeClassifierError, match="chronological"):
        classify_regime(dupe, as_of_utc=AS_OF)
