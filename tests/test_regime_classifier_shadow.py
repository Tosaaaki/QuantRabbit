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


def test_choppy_series_classifies_range() -> None:
    # Oscillation with wide-enough bands but low efficiency -> RANGE.
    closes = [1.1000 + (0.0030 if i % 2 else -0.0030) for i in range(140)]
    result = classify_regime(_candles(closes), as_of_utc=AS_OF)

    assert result["regime"] == "RANGE"
    assert result["components"]["efficiency_ratio"] < 0.35


def test_very_narrow_band_classifies_squeeze() -> None:
    closes = [1.1000 + (0.00002 if i % 2 else -0.00002) for i in range(140)]
    result = classify_regime(_candles(closes), as_of_utc=AS_OF)

    assert result["regime"] == "SQUEEZE"
    assert result["components"]["bb_width_fraction"] < 0.0025


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


def test_causality_and_shape_fail_closed() -> None:
    closes = [1.1000 + i * 0.0002 for i in range(140)]
    # A candle at/after the decision clock is non-causal.
    future = _candles(closes)
    future[-1]["time"] = AS_OF
    with pytest.raises(RegimeClassifierError, match="strictly before"):
        classify_regime(future, as_of_utc=AS_OF)

    with pytest.raises(RegimeClassifierError, match="insufficient"):
        classify_regime(_candles(closes[:50]), as_of_utc=AS_OF)

    dupe = _candles(closes)
    dupe[10]["time"] = dupe[9]["time"]
    with pytest.raises(RegimeClassifierError, match="chronological"):
        classify_regime(dupe, as_of_utc=AS_OF)
