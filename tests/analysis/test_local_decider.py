from __future__ import annotations

import math

from analysis.local_decider import heuristic_decision


def _base_perf() -> dict:
    return {
        "macro": {"pf": 1.02, "win_rate": 0.52, "sample": 80},
        "micro": {"pf": 1.01, "win_rate": 0.51, "sample": 80},
        "scalp": {"pf": 1.0, "win_rate": 0.5, "sample": 80},
    }


def _candles(*, n: int, drift: float) -> list[dict[str, float]]:
    base = 150.0
    out: list[dict[str, float]] = []
    for i in range(n):
        wave = 0.03 * math.sin(i / 3.0)
        close = base + drift * i + wave
        prev = out[-1]["close"] if out else close - drift
        open_price = prev
        high = max(open_price, close) + 0.01
        low = min(open_price, close) - 0.01
        out.append(
            {
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
            }
        )
    return out


def test_heuristic_decision_prefers_trend_when_forecast_is_directional() -> None:
    payload = {
        "event_soon": False,
        "factors_m1": {
            "ma10": 150.95,
            "ma20": 150.80,
            "adx": 30.0,
            "rsi": 62.0,
            "atr_pips": 8.4,
            "vol_5m": 1.5,
        },
        "factors_h4": {
            "ma10": 151.2,
            "ma20": 150.85,
            "adx": 33.0,
        },
        "perf": _base_perf(),
        "perf_hourly": {},
    }
    out = heuristic_decision(payload)
    forecast = out.get("forecast_bias")
    assert isinstance(forecast, dict)
    assert float(forecast.get("trend_strength") or 0.0) >= float(forecast.get("range_pressure") or 0.0)
    assert out["ranked_strategies"][0] in {
        "H1Momentum",
        "TrendMA",
        "TrendMomentumMicro",
        "MicroMomentumStack",
        "MomentumBurst",
    }


def test_heuristic_decision_prefers_reversion_when_range_pressure_is_high() -> None:
    payload = {
        "event_soon": False,
        "factors_m1": {
            "ma10": 150.00,
            "ma20": 150.01,
            "adx": 14.0,
            "rsi": 72.0,
            "atr_pips": 2.8,
            "vol_5m": 0.65,
        },
        "factors_h4": {
            "ma10": 150.10,
            "ma20": 150.10,
            "adx": 15.0,
        },
        "perf": _base_perf(),
        "perf_hourly": {},
    }
    out = heuristic_decision(payload)
    forecast = out.get("forecast_bias")
    assert isinstance(forecast, dict)
    assert float(forecast.get("range_pressure") or 0.0) >= float(forecast.get("trend_strength") or 0.0)
    assert out["ranked_strategies"][0] in {
        "BB_RSI",
        "BB_RSI_Fast",
        "MicroVWAPRevert",
        "MicroLevelReactor",
        "MicroPullbackEMA",
    }


def test_heuristic_decision_includes_projection_bias_when_candles_present() -> None:
    payload = {
        "event_soon": False,
        "factors_m1": {
            "ma10": 150.90,
            "ma20": 150.75,
            "adx": 28.0,
            "rsi": 60.0,
            "atr_pips": 6.4,
            "vol_5m": 1.3,
            "candles": _candles(n=80, drift=0.012),
        },
        "factors_h4": {
            "ma10": 151.30,
            "ma20": 150.95,
            "adx": 29.0,
            "candles": _candles(n=80, drift=0.085),
        },
        "perf": _base_perf(),
        "perf_hourly": {},
    }
    out = heuristic_decision(payload)
    forecast = out.get("forecast_bias")
    assert isinstance(forecast, dict)
    assert float(forecast.get("projection_score") or 0.0) > 0.0
    assert 0.0 <= float(forecast.get("projection_confidence") or 0.0) <= 1.0
