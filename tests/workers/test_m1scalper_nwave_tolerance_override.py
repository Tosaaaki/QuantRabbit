from __future__ import annotations

from types import SimpleNamespace


def _fac(close: float) -> dict:
    return {
        "candles": [
            {"open": 156.000, "high": 156.004, "low": 155.996, "close": 156.000},
            {"open": 156.000, "high": 156.004, "low": 155.996, "close": 156.000},
        ],
        "close": close,
        "ema20": 156.000,
        "ema10": 156.000,
        "rsi": 50.0,
        "atr": 0.01,
        "vol_5m": 0.2,
        "adx": 12.0,
        "bbw": 0.002,
    }


def _trend_flip_fac() -> dict:
    candles = [
        {"open": 150.000, "high": 150.020, "low": 149.995, "close": 150.010},
        {"open": 150.010, "high": 150.065, "low": 150.005, "close": 150.060},
    ]
    return {
        "candles": candles,
        "open": candles[-1]["open"],
        "high": candles[-1]["high"],
        "low": candles[-1]["low"],
        "close": candles[-1]["close"],
        "ema20": 150.000,
        "ema10": 150.032,
        "rsi": 60.0,
        "atr": 0.02,
        "atr_pips": 2.0,
        "vol_5m": 0.3,
        "adx": 24.0,
        "bbw": 0.0011,
    }


def test_nwave_tolerance_default_can_be_overridden_by_env(monkeypatch):
    from strategies.scalping import m1_scalper as strategy

    monkeypatch.setattr(strategy, "_load_scalper_config", lambda: {})
    monkeypatch.setattr(strategy, "_NWAVE_ALIGN_ENABLED", False)
    monkeypatch.setattr(
        strategy,
        "detect_latest_n_wave",
        lambda _candles: SimpleNamespace(
            direction="long",
            entry_price=156.000,
            pullback_pips=1.2,
            amplitude_pips=3.0,
            quality=1.0,
        ),
    )
    monkeypatch.delenv("M1SCALP_NWAVE_TOL_DEF_PIPS", raising=False)
    monkeypatch.delenv("M1SCALP_NWAVE_TOLERANCE_DEFAULT_PIPS", raising=False)

    # close is 0.45p above entry. default tolerance floor is 0.42p, so this is skipped as late.
    signal = strategy.M1Scalper.check(_fac(close=156.0045))
    assert signal is None

    monkeypatch.setenv("M1SCALP_NWAVE_TOL_DEF_PIPS", "0.50")
    signal = strategy.M1Scalper.check(_fac(close=156.0045))
    assert isinstance(signal, dict)
    assert signal.get("action") == "OPEN_LONG"
    assert float(signal.get("entry_tolerance_pips") or 0.0) >= 0.50


def test_trend_flip_extreme_guard_blocks_extended_trend_long(monkeypatch):
    from strategies.scalping import m1_scalper as strategy

    monkeypatch.setattr(strategy, "_load_scalper_config", lambda: {})
    monkeypatch.setattr(strategy, "_shock_guard", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(strategy, "_liquidity_guard", lambda: True)
    monkeypatch.setattr(strategy, "detect_latest_n_wave", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(strategy, "_BREAKOUT_RETEST_ENABLED", False)
    monkeypatch.setattr(strategy, "_VSHAPE_REBOUND_ENABLED", False)
    monkeypatch.setattr(strategy, "derive_pattern_signature", None)

    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_GUARD_ENABLED", False)
    signal = strategy.M1Scalper.check(_trend_flip_fac())
    assert isinstance(signal, dict)
    assert signal.get("tag") == "M1Scalper-trend-long"

    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_GUARD_ENABLED", True)
    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_ATR_MAX_PIPS", 3.2)
    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_BBW_MAX", 0.0014)
    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_GAP_MIN_PIPS", 2.5)
    monkeypatch.setattr(strategy, "_TREND_FLIP_EXTREME_LONG_RSI_MIN", 58.0)

    blocked = strategy.M1Scalper.check(_trend_flip_fac())
    assert blocked is None
