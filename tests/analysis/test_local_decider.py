from __future__ import annotations

from analysis.local_decider import heuristic_decision


def _base_perf() -> dict:
    return {
        "macro": {"pf": 1.02, "win_rate": 0.52, "sample": 80},
        "micro": {"pf": 1.01, "win_rate": 0.51, "sample": 80},
        "scalp": {"pf": 1.0, "win_rate": 0.5, "sample": 80},
    }


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
